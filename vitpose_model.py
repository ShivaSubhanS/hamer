from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from mmpose.apis import inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result

os.environ["PYOPENGL_PLATFORM"] = "egl"

# Monkey-patch the _inference_single_pose_model to add dataset_idx for ViTPose+ MoE
import mmpose.apis.inference as inference_module

# Store original function
_original_inference_single_pose_model = inference_module._inference_single_pose_model

def _patched_inference_single_pose_model(model, img_or_path, bboxes, dataset='TopDownCocoDataset', 
                                         dataset_info=None, return_heatmap=False, dataset_idx=4):
    """Patched version that adds dataset_idx support for ViTPose+ MoE models."""
    # Call original function but we need to inject dataset_idx into the data pipeline
    # We'll do this by patching the test_pipeline temporarily
    cfg = model.cfg
    original_pipeline = cfg.test_pipeline
    
    # Add dataset_idx injection to the pipeline
    # Find the LoadImageFromFile step and add dataset_idx after it
    import copy
    modified_pipeline = copy.deepcopy(original_pipeline)
    
    # Inject a custom transform that adds dataset_idx
    class AddDatasetIdx:
        def __init__(self, dataset_idx):
            self.dataset_idx = dataset_idx
        
        def __call__(self, results):
            results['dataset_idx'] = self.dataset_idx
            return results
    
    # Insert after LoadImageFromFile (usually first step)
    modified_pipeline.insert(1, AddDatasetIdx(dataset_idx))
    
    # Temporarily replace the pipeline
    cfg.test_pipeline = modified_pipeline
    
    try:
        result = _original_inference_single_pose_model(model, img_or_path, bboxes, dataset, dataset_info, return_heatmap)
    finally:
        # Restore original pipeline
        cfg.test_pipeline = original_pipeline
    
    return result

# Replace the function
inference_module._inference_single_pose_model = _patched_inference_single_pose_model

# project root directory
ROOT_DIR = "./"
VIT_DIR = os.path.join(ROOT_DIR, "third-party/ViTPose")

class ViTPoseModel(object):
    MODEL_DICT = {
        'ViTPose+-G (multi-task train, COCO)': {
            'config': f'{VIT_DIR}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_base_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py',
            'model': f'{ROOT_DIR}/_DATA/vitpose_ckpts/vitpose_base.pth',
        },
    }

    def __init__(self, device: str | torch.device):
        # Use the provided device (can be GPU now with base model)
        self.device = torch.device(device)
        self.model_name = 'ViTPose+-G (multi-task train, COCO)'
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        ckpt_path = dic['model']
        
        # Use the MoE config since vitpose_base.pth is a ViTPose+ MoE model
        from mmcv import Config
        cfg = Config.fromfile(dic['config'])
        cfg.load_from = 'local'
        
        # Set model to use only wholebody task (task_id=4) during inference
        # This tells the MoE model which expert head to use
        cfg.model.test_cfg['task_id'] = 4  # wholebody is task 4
        
        model = init_pose_model(cfg, ckpt_path, device=self.device)
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: list[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(image, out, kpt_score_threshold,
                                          vis_dot_radius, vis_line_thickness)
        return out, vis

    def predict_pose(
            self,
            image: np.ndarray,
            det_results: list[np.ndarray],
            box_score_threshold: float = 0.5) -> list[dict[str, np.ndarray]]:
        image = image[:, :, ::-1]  # RGB -> BGR
        person_results = process_mmdet_results(det_results, 1)
        
        # dataset_idx is now injected via monkey-patched _inference_single_pose_model
        out, _ = inference_top_down_pose_model(self.model,
                                               image,
                                               person_results=person_results,
                                               bbox_thr=box_score_threshold,
                                               format='xyxy')
        return out

    def visualize_pose_results(self,
                               image: np.ndarray,
                               pose_results: list[np.ndarray],
                               kpt_score_threshold: float = 0.3,
                               vis_dot_radius: int = 4,
                               vis_line_thickness: int = 1) -> np.ndarray:
        image = image[:, :, ::-1]  # RGB -> BGR
        vis = vis_pose_result(self.model,
                              image,
                              pose_results,
                              kpt_score_thr=kpt_score_threshold,
                              radius=vis_dot_radius,
                              thickness=vis_line_thickness)
        return vis[:, :, ::-1]  # BGR -> RGB