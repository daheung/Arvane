from source.depth_estimation.fine_depth import create_model_and_transforms
from utils import load_rgb

from source.depth_estimation.fine_depth import FDepthEstimation, EModelState

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class FEvalImgDesc:
    depth: np.ndarray
    fov_deg: torch.Tensor

class DepthEstimation(FDepthEstimation):
    def __init__(self, device=None):
        super(DepthEstimation, self).__init__(device)

        self.model = None
        self.transform = None
        self.state = None

    def initialize(self):
        model, transform = create_model_and_transforms()
        model.to(self.device)

        self.model, self.transform = model, transform

    def set_model_state(self, state: EModelState):
        if (self.model is None):
            raise RuntimeError('cannot setting model state before initializing process.')
        
        if (state == EModelState.TRAIN):
            self.model.train()
            self.state = EModelState.TRAIN

        if (state == EModelState.TEST):
            self.model.eval()
            self.state = EModelState.TEST
    
    def evaluation(self, image_path: list[str] | str) -> list[object]:
        if (self.model is None):
            raise RuntimeError('cannot evaluation model before initializing process.')

        if (self.state != EModelState.TEST):
            print(f'Warning: Depth-Estimation model is not evaluation state. force set evaluation state.')
            self.SetModelState(EModelState.TEST)
    
        ret_value = list()
        if (isinstance(image_path, str)):
            image_path = [image_path]
            
        for idx, path in enumerate(image_path):
            if (not os.path.isfile(path)):
                print(f'Can not predict model.')
                print(f'Image path is not file. Path: {path}')
                return
            
            image, _, f_px = load_rgb(path)
            print(f'Image Count: {idx + 1} - Path: {path} - f_px: {f_px}')
            depth = self.EvaluationImage(image, f_px)
            
            ret_value.append({
                'path': None,
                'image': image,
                'depth': {
                    'data': depth,
                    'depth_max': depth.max(),
                    'depth_min': depth.min(),
                }
            })
        
    
    def evaluation_image(self, image: np.ndarray, f_px: float) -> FEvalImgDesc:
        tf_image = self.transform(image).to(self.device)
        prediction = self.model.infer(tf_image, f_px=f_px)
        depth: np.ndarray = prediction["depth"].squeeze().cpu().numpy()
        fov_deg: torch.Tensor = prediction['fov_deg']
        return FEvalImgDesc(depth=depth, fov_deg=fov_deg)

    def convert_to_color_depth(self, depth: np.ndarray, dtype: np.dtype=np.uint8) -> np.ndarray:
        inverse_depth = 1 / depth
    
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
        )
    
        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(dtype)

        return color_depth
    
    def convert_to_gray_depth(
        self, 
        depth: np.ndarray,         
        depth_clip_rate: float | None = None,
        min_max_depth: tuple[int] | None = None,
        dtype: np.dtype=np.uint8,
    ) -> np.ndarray:
        conv_depth = depth
        if (depth_clip_rate is not None):
            depth_flat = depth.flatten()

            hist, bin_edges = np.histogram(depth_flat, bins=512, range=(0, depth_flat.max()))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            cumulative_hist = np.cumsum(hist)
            cumulative_hist_normalized = cumulative_hist / depth_flat.size

            lower_idx = np.searchsorted(cumulative_hist_normalized, 1 - depth_clip_rate)
            upper_idx = np.searchsorted(cumulative_hist_normalized, depth_clip_rate)

            clip_min = bin_centers[lower_idx]
            clip_max = bin_centers[upper_idx]

            conv_depth = np.clip(conv_depth, clip_min, clip_max)

        if (min_max_depth is not None):
            conv_depth = np.clip(conv_depth, min_max_depth[0], min_max_depth[1])

        return conv_depth.astype(dtype)

    def convert_to_rgb_depth(
        self, 
        image_desc: object, 
        depth_clip_rate: float | None = None,
        min_max_depth: tuple[int] | None = None
    ):
        try:
            color_depth = self.ConvertToColorDepth(image_desc['depth']['data'])
            gray_depth  = self.ConvertToGrayDepth (image_desc['depth']['data'], depth_clip_rate, min_max_depth, dtype=np.uint8)
            return np.array([color_depth, gray_depth])
        
        except Exception as error:
            print(f'cannot convert rgb to rgb-d.')
            print(f'ERROR: {error}')
            return np.array([], dtype=np.uint8)