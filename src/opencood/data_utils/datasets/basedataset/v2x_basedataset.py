# -*- coding: utf-8 -*-
# Author: Dereje Shenkut <dshenkut@andrew.cmu.edu>
# Based on https://github.com/yifanlu0227/HEAL/blob/main/opencood/data_utils/datasets/basedataset/opv2v_basedataset.py and https://github.com/ucla-mobility/V2X-Real/blob/main/opencood/data_utils/datasets/basedataset.py
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
from collections import OrderedDict

import cv2
import h5py

import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import json
import random
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.camera_utils import load_camera_data
from opencood.utils.transformation_utils import x1_to_x2
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils import SUPER_CLASS_MAP

class V2XREALBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.use_hdf5 = True
        
        self.dataset_mode = params['dataset_mode']
        assert self.dataset_mode in ['vc', 'ic', 'v2v', 'i2i']

        # Multi-class setup
        self.class_names = SUPER_CLASS_MAP.keys()
        self.build_inverse_super_class_map()
        self.build_class_name2int_map()

        # Pre and Postprocessing 
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], self.class_names, train)


        if 'wild_setting' in params:
            self.seed = params['wild_setting']['seed']
            # whether to add time delay
            self.async_flag = params['wild_setting']['async']
            self.async_mode = \
                'sim' if 'async_mode' not in params['wild_setting'] \
                    else params['wild_setting']['async_mode']
            self.async_overhead = params['wild_setting']['async_overhead']

            # localization error
            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']

            # transmission data size
            self.data_size = \
                params['wild_setting']['data_size'] \
                    if 'data_size' in params['wild_setting'] else 0
            self.transmission_speed = \
                params['wild_setting']['transmission_speed'] \
                    if 'transmission_speed' in params['wild_setting'] else 27
            self.backbone_delay = \
                params['wild_setting']['backbone_delay'] \
                    if 'backbone_delay' in params['wild_setting'] else 0

        else:
            self.async_flag = False
            self.async_overhead = 0  # ms
            self.async_mode = 'sim'
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0  # Mb (Megabits)
            self.transmission_speed = 27  # Mbps
            self.backbone_delay = 0  # ms


        # Data Augmentation
        if 'data_augment' in params: # late and early
            self.data_augmentor = DataAugmentor(params['data_augment'], train)
        else: # intermediate
            self.data_augmentor = None

        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
        self.root_dir = root_dir 
        
        # print("Dataset dir:", root_dir)

        if 'train_params' not in params or \
                'max_cav' not in params['train_params']:
            self.max_cav = 4
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        # self.load_depth_file = True if 'depth' in params['input_source'] else False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                            else self.generate_object_center_camera
        self.generate_object_center_single = self.generate_object_center # will it follows 'self.generate_object_center' when 'self.generate_object_center' change?

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # by default, we load lidar, camera and metadata. But users may
        # define additional inputs/tasks
        self.add_data_extension = \
            params['add_data_extension'] if 'add_data_extension' \
                                            in params else []

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False

        # first load all paths of different scenarios
        self.scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        
        # self.scenario_folders = scenario_folders
        if not self.train and self.dataset_mode != "v2v":
            self.scenario_folders = [scenario_folder for scenario_folder in
                                     self.scenario_folders if "2023-04-07" not in scenario_folder.split("/")[-1]]
            # print(self.scenario_folders)
        self.reinitialize()

    def build_inverse_super_class_map(self):
        self.INVERSE_SUPER_CLASS_MAP = {}
        for super_class_name in SUPER_CLASS_MAP.keys():
            for class_name in SUPER_CLASS_MAP[super_class_name]:
                self.INVERSE_SUPER_CLASS_MAP[class_name] = super_class_name

    def build_class_name2int_map(self):
        self.class_name2int = {}
        for i, class_name in enumerate(self.class_names):
            self.class_name2int[class_name] = i + 1

    def reinitialize(self):
        self.scenario_database = OrderedDict()
        self.len_record = []
        count = 0

        for (i, scenario_folder) in enumerate(self.scenario_folders):
            # Only include directories (not files) and sort them
            cav_list = sorted([x for x in os.listdir(scenario_folder) if os.path.isdir(os.path.join(scenario_folder, x))])
            # at least 1 cav should show up
            if self.train:
                random.shuffle(cav_list)
            else:
                # Reorder the cav_list based on dataset_mode:
                # vehicles (1,2) andinfrastructure (-2, -1)
                # vc (vehicle-centric): 
                if self.dataset_mode == 'vc':
                    cav_list = [idx for idx in cav_list if int(idx) >= 0] + \
                               [idx for idx in cav_list if int(idx) < 0]
                # v2v (vehicle-to-vehicle): only vehicles
                elif self.dataset_mode == 'v2v':
                    cav_list = [idx for idx in cav_list if int(idx) >= 0]
                # ic (infrastructure-centric): infrastructure first, then vehicles  
                elif self.dataset_mode == 'ic':
                    cav_list = [idx for idx in cav_list if int(idx) < 0] + \
                               [idx for idx in cav_list if int(idx) >= 0]
                # i2i (infrastructure-to-infrastructure): only infrastructure
                elif self.dataset_mode == 'i2i':
                    cav_list = [idx for idx in cav_list if int(idx) < 0]
                else:
                    raise ValueError(f"{self.dataset_mode} must be either 'vc', 'v2v', or 'vc'")
                
                # Skip if no CAVs left after filtering
                if len(cav_list) == 0:
                    continue
            i = count
            count += 1
            self.scenario_database.update({i: OrderedDict()})
            # at least 1 cav should show up
            # print(cav_list)
            assert len(cav_list) > 0

            """
            roadside unit data's id is always negative, so here we want to
            make sure they will be in the end of the list as they shouldn't
            be ego vehicle.
            """
            # if int(cav_list[0]) < 0:
            #     cav_list = cav_list[1:] + [cav_list[0]]

            """
            make the first cav to be ego modality
            """
            if getattr(self, "heterogeneous", False):
                scenario_name = scenario_folder.split("/")[-1]
                cav_list = self.adaptor.reorder_cav_list(cav_list, scenario_name)


            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs reinitialize')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' not in x])
                
                # this timestamp is not ready
                # yaml_files = [x for x in yaml_files if not ("2021_08_20_21_10_24" in x and "000265" in x)]

                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()
                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.bin')
                    camera_files = self.find_camera_files(cav_path, 
                                                timestamp)
                    # depth_files = self.find_camera_files(cav_path, 
                    #                             timestamp, sensor="depth")
                    # depth_files = [depth_file.replace("OPV2V", "OPV2V_Hetero") for depth_file in depth_files]

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                    self.scenario_database[i][cav_id][timestamp]['cameras'] = \
                        camera_files
                    # self.scenario_database[i][cav_id][timestamp]['depths'] = \
                    #     depth_files

                    if getattr(self, "heterogeneous", False):
                        # breakpoint()
                        # Default modality based on agent type
                        default_modality = 'm2' if int(cav_id) < 0 else 'm1'
                        
                        # Get modality with fallback to default
                        scenario_name = scenario_folder.split("/")[-1]
                        try:
                            if self.modality_assignment is not None and \
                               scenario_name in self.modality_assignment and \
                               cav_id in self.modality_assignment[scenario_name]:
                                cav_modality = self.adaptor.reassign_cav_modality(
                                    self.modality_assignment[scenario_name][cav_id], j)
                            else:
                                cav_modality = default_modality
                        except:
                            cav_modality = default_modality

                        self.scenario_database[i][cav_id][timestamp]['modality_name'] = cav_modality
                        self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                            self.adaptor.switch_lidar_channels(cav_modality, lidar_file)


                   # load extra data
                    for file_extension in self.add_data_extension:
                        file_name = \
                            os.path.join(cav_path,
                                         timestamp + '_' + file_extension)

                        self.scenario_database[i][cav_id][timestamp][
                            file_extension] = file_name                  

                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the 
                # scene.
                if j == 0:
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False
        print("len:", self.len_record[-1])

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.
        
        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix. If set to false, meaning when other cavs
            project their LiDAR point cloud to ego, they are projecting to
            past ego pose.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
        # ego_cav_content = self.calc_dist_to_ego(scenario_database, timestamp_key)
        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']
            # timestamp_delay = \
            #     self.time_delay_calculation(cav_content['ego'])

            # if timestamp_index - timestamp_delay <= 0:
            #     timestamp_delay = timestamp_index
            # timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            # timestamp_key_delay = self.return_timestamp_key(scenario_database,
            #                                                 timestamp_index_delay)
            # # add time delay to vehicle parameters
            # data[cav_id]['time_delay'] = timestamp_delay
            # load the corresponding data into the dictionary
            # data[cav_id]['params'] = self.reform_param(cav_content,
            #                                            ego_cav_content,
            #                                            timestamp_key,
            #                                            timestamp_key_delay,
            #                                            cur_ego_pose_flag)
            # try:
            # load param file: json is faster than yaml
            json_file = cav_content[timestamp_key]['yaml'].replace("yaml", "json")
            
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data[cav_id]['params'] = json.load(f)
            else:
                data[cav_id]['params'] = \
                    load_yaml(cav_content[timestamp_key]['yaml'])
            # except:
            # print(cav_id, cav_content)


            # load camera file: hdf5 is faster than png
            hdf5_file = cav_content[timestamp_key]['cameras'][0].replace("cam1.jpeg", "imgs.hdf5")

            if self.use_hdf5 and os.path.exists(hdf5_file):
                with h5py.File(hdf5_file, "r") as f:
                    data[cav_id]['camera_data'] = []
                    for i in range(4):
                        if self.load_camera_file:
                            data[cav_id]['camera_data'].append(Image.fromarray(f[f'camera{i}'][()]))


                    # Determine number of cameras based on agent type
                    # num_cameras = 2 if int(cav_id) < 0 else 4  # 2 for infra, 4 for vehicles
                    
                    # for i in range(num_cameras):
                    #     if self.load_camera_file:
                    #         data[cav_id]['camera_data'].append(Image.fromarray(f[f'cam{i+1}'][()]))
            else:
                if self.load_camera_file:
                    # Get camera files based on agent type
                    # if int(cav_id) < 0:  # Infrastructure
                    #     camera_files = [
                    #         cav_content[timestamp_key]['cameras'][0],  # cam1
                    #         cav_content[timestamp_key]['cameras'][1]   # cam2
                    #     ]
                    # else:  # Vehicle
                    #     camera_files = cav_content[timestamp_key]['cameras']
                    
                    data[cav_id]['camera_data'] = load_camera_data(cav_content[timestamp_key]['cameras'])
                # if self.load_depth_file:
                #     data[cav_id]['depth_data'] = \
                #         load_camera_data(cav_content[timestamp_key]['depths']) 


            # load lidar file
            if self.load_lidar_file or self.visualize:
                data[cav_id]['lidar_np'] = \
                    pcd_utils.load_lidar_bin(cav_content[timestamp_key]['lidar'], zero_intensity=True)

            if getattr(self, "heterogeneous", False):
                data[cav_id]['modality_name'] = cav_content[timestamp_key]['modality_name']

            for file_extension in self.add_data_extension:
                # if not find in the current directory
                # go to additional folder
                if not os.path.exists(cav_content[timestamp_key][file_extension]):
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace("train","additional/train")
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace("validate","additional/validate")
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace("test","additional/test")
                    
                if '.yaml' in file_extension:
                    data[cav_id][file_extension] = \
                        load_yaml(cav_content[timestamp_key][file_extension])
                else:
                    data[cav_id][file_extension] = \
                        cv2.imread(cav_content[timestamp_key][file_extension])

        # print(" DONE BASE retrieve_base_data")
        return data

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass
    def time_delay_calculation(self, ego_flag):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : int
            The time delay quantization.
        """
        # there is not time delay for ego vehicle
        if ego_flag:
            return 0
        # time delay real mode
        if self.async_mode == 'real':
            # in the real mode, time delay = systematic async time + data
            # transmission time + backbone computation time
            overhead_noise = np.random.uniform(0, self.async_overhead)
            tc = self.data_size / self.transmission_speed * 1000
            time_delay = int(overhead_noise + tc + self.backbone_delay)
        elif self.async_mode == 'sim':
            # in the simulation mode, the time delay is constant
            time_delay = np.abs(self.async_overhead)

        # the data is 10 hz for both opv2v and v2x-set
        # todo: it may not be true for other dataset like DAIR-V2X and V2X-Sim
        time_delay = time_delay // 100
        return time_delay if self.async_flag else 0
    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    def calc_dist_to_ego(self, scenario_database, timestamp_key):
        """
        Calculate the distance to ego for each cav.
        """
        ego_lidar_pose = None
        ego_cav_content = None
        # Find ego pose first
        for cav_id, cav_content in scenario_database.items():
            if cav_content['ego']:
                ego_cav_content = cav_content
                ego_lidar_pose = \
                    load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
                break

        assert ego_lidar_pose is not None

    def add_loc_noise(self, pose, xyz_std, ryp_std):
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : list
            x,y,z,roll,yaw,pitch

        xyz_std : float
            std of the gaussian noise on xyz

        ryp_std : float
            std of the gaussian noise
        """
        if not self.train:
            np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, xyz_std, 3)
        ryp_std = np.random.normal(0, ryp_std, 3)
        noise_pose = [pose[0] + xyz_noise[0],
                      pose[1] + xyz_noise[1],
                      pose[2] + xyz_noise[2],
                      pose[3],
                      pose[4] + ryp_std[1],
                      pose[5]]
        return noise_pose
  
 
    def filter_boxes_by_class(self, object_dict):
        filtered_object_dict = OrderedDict()
        for obj_id, obj in object_dict.items():
            if obj['obj_type'].lower() in self.class_names:
                # Map class name (string) to int (np.array with shape (1,))
                obj['obj_type'] = np.array(
                    [self.class_name2int[obj['obj_type'].lower()]])
                filtered_object_dict[obj_id] = obj
        return filtered_object_dict
 
    # @staticmethod
    @staticmethod
    def find_camera_files(cav_path, timestamp, sensor="camera"):
        camera1_file = os.path.join(cav_path, timestamp + '_cam1.jpeg')
        camera2_file = os.path.join(cav_path, timestamp + '_cam2.jpeg')

        
        is_infrastructure = "/infrastructure/" in cav_path or "/-" in cav_path
        if is_infrastructure:
            # For infrastructure, duplicate the existing cameras
            return [camera1_file, camera2_file, camera1_file, camera2_file]
        else:
            camera3_file = os.path.join(cav_path, timestamp + '_cam3.jpeg')
            camera4_file = os.path.join(cav_path, timestamp + '_cam4.jpeg')
            return [camera1_file, camera2_file, camera3_file, camera4_file]

    def map_class_name_to_super_class_name(self, object_dict):
        new_object_dict = OrderedDict()
        for obj_id, obj in object_dict.items():
            if obj['obj_type'] not in self.INVERSE_SUPER_CLASS_MAP:
                continue
            obj['obj_type'] = self.INVERSE_SUPER_CLASS_MAP[obj['obj_type']]
            new_object_dict[obj_id] = obj
        return new_object_dict

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask


    def reform_param(self, cav_content, ego_content, timestamp_cur,
                     timestamp_delay, cur_ego_pose_flag):
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose for other CAVs.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.

        ego_content : dict
            Ego vehicle content.

        timestamp_cur : str
            The current timestamp.

        timestamp_delay : str
            The delayed timestamp.

        cur_ego_pose_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        Return
        ------
        The merged parameters.
        """
        cur_params = load_yaml(cav_content[timestamp_cur]['yaml'])
        delay_params = load_yaml(cav_content[timestamp_delay]['yaml'])

        cur_ego_params = load_yaml(ego_content[timestamp_cur]['yaml'])
        delay_ego_params = load_yaml(ego_content[timestamp_delay]['yaml'])

        # we need to calculate the transformation matrix from cav to ego
        # at the delayed timestamp
        delay_cav_lidar_pose = delay_params['lidar_pose']
        delay_ego_lidar_pose = delay_ego_params["lidar_pose"]

        cur_ego_lidar_pose = cur_ego_params['lidar_pose']
        cur_cav_lidar_pose = cur_params['lidar_pose']

        if not cav_content['ego'] and self.loc_err_flag:
            delay_cav_lidar_pose = self.add_loc_noise(delay_cav_lidar_pose,
                                                      self.xyz_noise_std,
                                                      self.ryp_noise_std)
            cur_cav_lidar_pose = self.add_loc_noise(cur_cav_lidar_pose,
                                                    self.xyz_noise_std,
                                                    self.ryp_noise_std)

        if cur_ego_pose_flag:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             delay_ego_lidar_pose)
            spatial_correction_matrix = x1_to_x2(delay_ego_lidar_pose,
                                                 cur_ego_lidar_pose)
        # This is only used for late fusion, as it did the transformation
        # in the postprocess, so we want the gt object transformation use
        # the correct one
        gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose,
                                            cur_ego_lidar_pose)
        # Map each category class name to a supper class name; Group similar classes to a super class
        self.map_class_name_to_super_class_name(cur_params['vehicles'])

        # we always use current timestamp's gt bbx to gain a fair evaluation
        delay_params['vehicles'] = self.filter_boxes_by_class(
            cur_params['vehicles'])
        delay_params['transformation_matrix'] = transformation_matrix
        delay_params['gt_transformation_matrix'] = \
            gt_transformation_matrix
        delay_params['spatial_correction_matrix'] = spatial_correction_matrix

        return delay_params

    def generate_object_center_lidar(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_object_center(cav_contents,
                                                        reference_lidar_pose)

    def generate_object_center_camera(self, 
                                cav_contents, 
                                reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.
        
        visibility_map : np.ndarray
            for OPV2V, its 256*256 resolution. 0.39m per pixel. heading up.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_visible_object_center(
            cav_contents, reference_lidar_pose
        )

    def get_ext_int(self, params, camera_id):

        # print(params.keys())
        # print(params)
        # import pdb; pdb.set_trace()
        # print(camera_id)
        # if 'cam3' in params.keys():
        # print(camera_id)
        try:
            camera_coords = np.array(params["cam%d" % (camera_id+1)]["cords"]).astype(
                np.float32)
            camera_to_lidar = x1_to_x2(
                camera_coords, params["lidar_pose_clean"]
            ).astype(np.float32)  # T_LiDAR_camera
            camera_to_lidar = camera_to_lidar @ np.array(
                [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                dtype=np.float32)  # UE4 coord to opencv coord
            camera_intrinsic = np.array(params["cam%d" % (camera_id+1)]["intrinsic"]).astype(
                np.float32
            )
            return camera_to_lidar, camera_intrinsic
        except:
            camera_coords = np.array(params["cam%d" % ((camera_id+1)//2)]["cords"]).astype(
                np.float32)
            camera_to_lidar = x1_to_x2(
                camera_coords, params["lidar_pose_clean"]
            ).astype(np.float32)  # T_LiDAR_camera
            camera_to_lidar = camera_to_lidar @ np.array(
                [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                dtype=np.float32)  # UE4 coord to opencv coord
            camera_intrinsic = np.array(params["cam%d"  % ((camera_id+1)//2)]["intrinsic"]).astype(
                np.float32
            )
            return camera_to_lidar, camera_intrinsic