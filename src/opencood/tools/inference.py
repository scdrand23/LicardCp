# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import time
from typing import OrderedDict
import importlib
import matplotlib
matplotlib.use('Agg')
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
torch.multiprocessing.set_sharing_strategy('file_system')

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="102.4,102.4",
                        help="detection range is [-102.4, +102.4, -102.4, +102.4]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    # print("Starting inference with options:", opt)
    num_samples = 500  # Change this number to control how many samples to process

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)
    # print("Loaded hypes yaml config")

    if 'heter' in hypes:
        # print("Processing heterogeneous config")
        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]
        # print("New CAV range:", new_cav_range)

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)
        # print("Updated hypes with new ranges")
        
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    # print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    # print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")

    # print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    # print(f"Resume from epoch {resume_epoch}")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)
    # print("Set random seed to 303")
    
    # build dataset for each noise setting
    # print('Building Dataset...')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    
    # Create subset of dataset
    subset_indices = list(range(min(num_samples, len(opencood_dataset))))
    subset_dataset = Subset(opencood_dataset, subset_indices)
    # print(f"Using subset of dataset: {len(subset_dataset)} / {len(opencood_dataset)} samples")
    
    data_loader = DataLoader(opencood_dataset,  # Changed from opencood_dataset to subset_dataset
                            batch_size=1,
                            num_workers=2,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    # breakpoint()
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    
    infer_info = opt.fusion_method + opt.note
    # print(f"Starting inference with info: {infer_info}")

    for i, batch_data in enumerate(data_loader):
        # print(f"Processing batch {i} with info: {infer_info}")
        # breakpoint()
        if batch_data is None:
            # print(f"Skipping batch {i} - no data")
            continue
            
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            # print(f"Batch {i} moved to device {device}")

            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                # breakpoint()
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')
            # print(f"Completed inference for batch {i} using {opt.fusion_method} fusion")

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            
            # print(f"Batch {i} - Pred boxes: {pred_box_tensor.shape if pred_box_tensor is not None else 'None'}")
            # print(f"Batch {i} - GT boxes: {gt_box_tensor.shape if gt_box_tensor is not None else 'None'}")
            
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                    print(f"Created directory for NPY files: {npy_save_path}")
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)
                print(f"Saved NPY files for batch {i}")

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "agent_modality_list": agent_modality_list})
                print(f"Updated heterogeneous info for batch {i}")

            # if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
            #     vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
            #     if not os.path.exists(vis_save_path_root):
            #         os.makedirs(vis_save_path_root)
            #         print(f"Created visualization directory: {vis_save_path_root}")
                 
            #     vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
            #     #                camera_params = None
            #     # image_np = None
            #     # ego_modality_name = None
            #     # camera_input_key = None
            #     #  Example usage
            #     # simple_vis.visualize_on_image(
            #     #     pred_box_tensor,
            #     #     gt_box_tensor, 
            #     #     None,
            #     #     save_path='output.png',
            #     #     camera_params={
            #     #         'canvas_shape': (None, None),
            #     #         'camera_center_coords': (-15, 0, 10),
            #     #         'camera_focus_coords': (-15 + 0.9396926, 0, 10 - 0.44202014),
            #     #         'focal_length': None  # Will use default calculation
            #     #     },
            #     #     left_hand=False
            #     # )
            #     simple_vis.visualize(infer_result,
            #                         batch_data['ego'][
            #                             'origin_lidar'][0],
            #                         hypes['postprocess']['gt_range'],
            #                         vis_save_path,
            #                         method='3d',
            #                         left_hand=left_hand)
            #     print(f"Saved visualization for batch {i} at {vis_save_path}")
                
        torch.cuda.empty_cache()
        # print(f"Cleared CUDA cache after batch {i}")

    print("Computing final evaluation metrics...")
    _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, infer_info)
    print(f"Completed evaluation - AP50: {ap50:.4f}, AP70: {ap70:.4f}")

if __name__ == '__main__':
    main()
