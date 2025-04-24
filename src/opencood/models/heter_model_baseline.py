# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# A unified framework for LiDAR-only / Camera-only / Heterogeneous collaboration.
# Support multiple fusion strategies.


import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, CoBEVT, Where2commFusion, Who2comFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import torchvision

class HeterModelBaseline(nn.Module):
    def __init__(self, args):
        super(HeterModelBaseline, self).__init__()
        self.args = args
        print(self.args.keys())
        modality_name_list = list(args.keys())
        # self.num_class = args['num_class'] if args['num_class'] is not None else 1
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list

        self.ego_modality = args['ego_modality']

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            """
            Encoder building
            """
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """
            Backbone building 
            """
            setattr(self, f"backbone_{modality_name}", BaseBEVBackbone(model_setting['backbone_args'], 
                                                                       model_setting['backbone_args'].get('inplanes',64)))

            """
            shrink conv building
            """
            setattr(self, f"shrinker_{modality_name}", DownsampleConv(model_setting['shrink_header']))

            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        self.supervise_single = False
        if args.get("supervise_single", False):
            self.supervise_single = True
            in_head_single = args['in_head_single']
            setattr(self, f'cls_head_single', nn.Conv2d(in_head_single, args['anchor_number']*self.num_class*self.num_class, kernel_size=1))
            setattr(self, f'reg_head_single', nn.Conv2d(in_head_single, args['anchor_number'] * 7*self.num_class, kernel_size=1))
            setattr(self, f'dir_head_single', nn.Conv2d(in_head_single, args['anchor_number'] *  args['dir_args']['num_bins'], kernel_size=1))


        if args['fusion_method'] == "max":
            self.fusion_net = MaxFusion()
        if args['fusion_method'] == "att":
            self.fusion_net = AttFusion(args['att']['feat_dim'])
        if args['fusion_method'] == "disconet":
            self.fusion_net = DiscoFusion(args['disconet']['feat_dim'])
        if args['fusion_method'] == "v2vnet":
            self.fusion_net = V2VNetFusion(args['v2vnet'])
        if args['fusion_method'] == 'v2xvit':
            self.fusion_net = V2XViTFusion(args['v2xvit'])
        if args['fusion_method'] == 'cobevt':
            self.fusion_net = CoBEVT(args['cobevt'])
        if args['fusion_method'] == 'where2comm':
            self.fusion_net = Where2commFusion(args['where2comm'])
        if args['fusion_method'] == 'who2com':
            self.fusion_net = Who2comFusion(args['who2com'])


        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        """
        Shared Heads
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], 
                         args['dir_args']['num_bins'] * args['anchor_number'] ,
                         kernel_size=1)
        
        # compressor will be only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])
            self.model_train_init()


        # check again which module is not fixed.
        check_trainable_module(self)

    def model_train_init(self):
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def forward(self, data_dict):
        output_dict = {}
        # print("\n=== Starting forward pass ===")
        # print("Input data_dict keys:", data_dict.keys())
        # breakpoint()
        # Step 1: Get agent modalities and setup
        agent_modality_list = data_dict['agent_modality_list']
        # print("\nStep 1: Agent modalities:", agent_modality_list)
        
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len']
        # print("Record length:", record_len)
        # print("Affine matrix shape:", affine_matrix.shape)

        # Step 2: Count modalities
        modality_count_dict = Counter(agent_modality_list)
        # print("\nStep 2: Modality counts:", dict(modality_count_dict))
        
        # Step 3: Extract features for each modality
        # print("\nStep 3: Extracting features for each modality")
        modality_feature_dict = {}  
        # breakpoint()
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                # print(f"Skipping {modality_name} - not present")
                continue
                
            # print(f"\nProcessing {modality_name}:")
            # print(data_dict.keys())
            # import pdb; pdb.set_trace()
            encoder = getattr(self, f"encoder_{modality_name}")
            encoder_class_name = encoder.__class__.__name__
            # print(f"Encoder class name: {encoder_class_name}")

            model_setting = self.args[modality_name]
            core_method = model_setting['core_method']
            # print(f"Core method from settings: {core_method}")

            encoder = getattr(self, f"encoder_{modality_name}")
            # print(f"Full encoder path: {encoder.__module__}.{encoder.__class__.__name__}")

            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            # print(f"Encoder output shape: {feature.shape}")
            
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            # print(f"Backbone output shape: {feature.shape}")
            
            feature = eval(f"self.shrinker_{modality_name}")(feature)
            # print(f"Shrinker output shape: {feature.shape}")
            
            modality_feature_dict[modality_name] = feature

        # Step 4: Process camera features if present
        # print("\nStep 4: Processing camera features")
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # print(f"\nProcessing camera modality: {modality_name}")
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))
                    # print(f"Original size: {H}x{W}, Target size: {target_H}x{target_W}")

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    # print(f"After cropping shape: {modality_feature_dict[modality_name].shape}")
                    
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })
                        # print(f"Added depth supervision items for {modality_name}")

        # Step 5: Assemble heterogeneous features
        # print("\nStep 5: Assembling heterogeneous features")
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        
        # print("Feature shapes by modality:")
        # for k,v in modality_feature_dict.items():
        #     print(f"{k}: {v.shape}")
        # print("Processing agent modalities:", agent_modality_list)
        
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            if feat_idx >= modality_feature_dict[modality_name].shape[0]:
                # print(f"Warning: Index {feat_idx} out of bounds for {modality_name} with shape {modality_feature_dict[modality_name].shape}")
                continue
            feature = modality_feature_dict[modality_name][feat_idx]
            # print(f"Adding feature for {modality_name}[{feat_idx}] shape: {feature.shape}")
            heter_feature_2d_list.append(feature)
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)
        # print("Final stacked feature shape:", heter_feature_2d.shape)
        
        # Step 6: Compression if enabled
        if self.compress:
            # print("\nStep 6: Compressing features")
            heter_feature_2d = self.compressor(heter_feature_2d)
            # print("Compressed feature shape:", heter_feature_2d.shape)

        # Step 7: Single supervision if enabled
        # print("\nStep 7: Single supervision")
        if self.supervise_single:
            # print("Performing single supervision")
            cls_preds_before_fusion = self.cls_head_single(heter_feature_2d)
            reg_preds_before_fusion = self.reg_head_single(heter_feature_2d)
            dir_preds_before_fusion = self.dir_head_single(heter_feature_2d)
            output_dict.update({'cls_preds_single': cls_preds_before_fusion,
                                'reg_preds_single': reg_preds_before_fusion,
                                'dir_preds_single': dir_preds_before_fusion})
            # print("Added single supervision predictions to output")

        # Step 8: Feature fusion and final predictions
        # print("\nStep 8: Feature fusion and predictions")
        fused_feature = self.fusion_net(heter_feature_2d, record_len, affine_matrix)
        # print("Fused feature shape:", fused_feature.shape)

        if self.shrink_flag:
            # print("Applying shrink convolution")
            fused_feature = self.shrink_conv(fused_feature)
            # print("After shrink shape:", fused_feature.shape)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)
        
        # print("\nFinal prediction shapes:")
        # print(f"Classification: {cls_preds.shape}")
        # print(f"Regression: {reg_preds.shape}")
        # print(f"Direction: {dir_preds.shape}")

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})

        # print("\n=== Forward pass complete ===\n")
        return output_dict
