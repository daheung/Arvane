import os
import sys
import box
import time
import tqdm
import torch
import logging
import trimesh
import numpy as np

from typing import List, Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Loading ReconPredictor...")

from source.predictor.recon import data
from source.predictor.recon import utils
from source.predictor.recon import network
from source.predictor.utils import pad_last

from source.runtime.infer.store import ReconLog

from dataclasses import dataclass, field
from typing import Optional    

@dataclass(slots=True)
class ReconInferArray:
    images : List = field(default_factory=list)
    depths : List = field(default_factory=list)
    k_color: List = field(default_factory=list)
    k_depth: List = field(default_factory=list)
    poses  : List = field(default_factory=list)

    M                     : torch.Tensor = None
    running_count         : torch.Tensor = None
    running_density       : torch.Tensor = None
    running_tsdf          : torch.Tensor = None
    global_coords         : torch.Tensor = None
    running_density_weight: torch.Tensor = None
    running_tsdf_weight   : torch.Tensor = None

@dataclass(slots=True)
class ReconSession:
    user_id    : str
    task_id    : str
    out_path   : Optional[str] = None

    infer      : ReconInferArray = field(default_factory=ReconInferArray)
    log        : ReconLog        = field(default_factory=ReconLog)
    
class ReconPro(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        img_feature_dim = 47
        self.cnn2d = network.Cnn2d(out_dim=img_feature_dim)
        self.fusion = network.FeatureFusion(in_c=img_feature_dim)
        self.voxel_feat_dim = self.fusion.out_c

        self.depth_guidance = self.config.depth_guidance

        if self.depth_guidance.enabled:
            if self.depth_guidance.density_fusion_channel:
                self.voxel_feat_dim += 1
            elif self.depth_guidance.tsdf_fusion_channel:
                self.voxel_feat_dim += 1

        self.cnn3d = network.Cnn3d(in_c=self.voxel_feat_dim)

        if self.config.point_backprojection:
            self.cnn2d_pb_out_dim = img_feature_dim
            self.cnn2d_pb = network.Cnn2d(out_dim=self.cnn2d_pb_out_dim)
            self.point_feat_mlp = torch.nn.Sequential(
                network.ResBlock1d(self.cnn2d_pb_out_dim),
                network.ResBlock1d(self.cnn2d_pb_out_dim),
            )
            self.point_fusion = network.FeatureFusion(in_c=self.cnn2d_pb_out_dim)

        surface_pred_input_dim = occ_pred_input_dim = self.cnn3d.out_c
        if self.config.point_backprojection:
            surface_pred_input_dim += self.cnn2d_pb_out_dim

        if self.depth_guidance.enabled:
            if self.config.point_backprojection:
                if self.depth_guidance.density_fusion_channel:
                    surface_pred_input_dim += 1
                elif self.depth_guidance.tsdf_fusion_channel:
                    surface_pred_input_dim += 1

        self.surface_predictor = torch.nn.Sequential(
            torch.nn.Conv1d(surface_pred_input_dim, 32, 1),
            network.ResBlock1d(32),
            network.ResBlock1d(32),
            torch.nn.Conv1d(32, 1, 1),
        )
        self.occ_predictor = torch.nn.Sequential(
            torch.nn.Conv1d(occ_pred_input_dim, 32, 1),
            network.ResBlock1d(32),
            network.ResBlock1d(32),
            torch.nn.Conv1d(32, 1, 1),
        )
    
    def get_img_voxel_feats_by_depth_guided_bp(
        self,
        rgb_imgs,
        pred_depth_imgs,
        poses,
        K_color,
        K_pred_depth,
        input_coords,
        use_highres_cnn=False,
        img_feats=None,
    ):
        img_voxel_feats, img_voxel_valid = self.get_img_voxel_feats_by_img_bp(
            rgb_imgs,
            poses,
            K_color,
            input_coords,
            use_highres_cnn=use_highres_cnn,
            img_feats=img_feats,
        )

        depth, depth_valid, z = utils.sample_posed_images(
            pred_depth_imgs[:, :, None],
            poses,
            K_pred_depth,
            input_coords,
            mode="nearest",
            return_z=True,
        )
        depth = depth.squeeze(2)

        depth_valid.masked_fill_(depth == 0, False)

        if "gaussian" in self.depth_guidance.bp_weighting:
            dist = (z - depth).abs()
            if self.depth_guidance.bp_weighting == "gaussian_12cm":
                weight = torch.exp(-((dist * 16) ** 2))
            elif self.depth_guidance.bp_weighting == "gaussian_24cm":
                weight = torch.exp(-((dist * 8) ** 2))
            else:
                raise NotImplementedError
            weight.masked_fill_(~depth_valid, 0)
            img_voxel_feats *= weight[:, :, None]

        elif "truncation" in self.depth_guidance.bp_weighting:
            dist = (z - depth).abs()
            if self.depth_guidance.bp_weighting == "truncation_3.5cm":
                weight = (dist < 0.035).float()
            elif self.depth_guidance.bp_weighting == "truncation_12cm":
                weight = (dist < 0.12).float()
            elif self.depth_guidance.bp_weighting == "truncation_24cm":
                weight = (dist < 0.24).float()
            elif self.depth_guidance.bp_weighting == "truncation_48cm":
                weight = (dist < 0.48).float()
            else:
                raise NotImplementedError

            weight.masked_fill_(~depth_valid, 0)
            img_voxel_feats *= weight[:, :, None]

        elif self.depth_guidance.bp_weighting == "none":
            ...
        else:
            raise NotImplementedError

        img_voxel_feats.masked_fill_(~img_voxel_valid[:, :, None], 0)

        return img_voxel_feats, img_voxel_valid

    def get_img_voxel_feats_by_img_bp(
        self,
        rgb_imgs,
        poses,
        K_color,
        input_coords,
        use_highres_cnn=False,
        img_feats=None,
    ):
        batch_size, n_imgs, _, imheight, imwidth = rgb_imgs.shape
        imsize = (imheight, imwidth)

        if img_feats is None:
            if use_highres_cnn:
                img_feats = self.cnn2d_pb(
                    rgb_imgs.view(batch_size * n_imgs, 3, imheight, imwidth)
                )
            else:
                img_feats = self.cnn2d(
                    rgb_imgs.view(batch_size * n_imgs, 3, imheight, imwidth)
                )

        img_feats = img_feats.view(batch_size, n_imgs, *img_feats.shape[1:])

        img_voxel_feats, img_voxel_valid = utils.sample_voxel_feats(
            img_feats, poses, K_color, input_coords, imsize
        )

        if (not self.training) and use_highres_cnn:
            # down-weight the high-res BP image features near the image border
            # to reduce boundary artifacts.
            # works at inference time, not tested with training

            xyz = input_coords
            batch_size = xyz.shape[0]
            xyz = xyz.view(batch_size, 1, -1, 3).transpose(3, 2)
            xyz = torch.cat((xyz, torch.ones_like(xyz[:, :, :1])), dim=2)

            featheight, featwidth = img_feats.shape[-2:]

            K = K_color.clone()
            K[:, :, 0] *= featwidth / imwidth
            K[:, :, 1] *= featheight / imheight

            device: torch.device = self.config.device
            with torch.autocast(enabled=False, device_type=device.type):
                xyz_cam = (torch.inverse(poses.float()) @ xyz)[:, :, :3]
                uv = K @ xyz_cam
            uv = uv[:, :, :2] / uv[:, :, 2:]

            featsize = torch.tensor(
                [featwidth, featheight], device=self.config.device, dtype=uv.dtype
            )[None, None, :, None]
            uv[:, :, 0].clamp_(0, imwidth)
            uv[:, :, 1].clamp_(0, imheight)
            border_dist = ((uv / featsize).round() * featsize - uv).abs().min(dim=2)[0]
            pixel_margin = 20
            weight = (border_dist / pixel_margin).clamp(0, 1)
            weight = torch.sigmoid(weight * 12 - 6)
            img_voxel_feats *= weight[:, :, None]

        return img_voxel_feats, img_voxel_valid

    def sample_point_features_by_linear_interp(
        self, coords, voxel_feats, voxel_valid, grid_origin
    ):
        """
        coords: BN3
        voxel_feats: BFXYZ
        voxel_valid: BXYZ
        grid_origin: B3
        """
        crop_size_m = (
            torch.tensor(voxel_feats.shape[2:], device=self.config.device)
            * self.config.voxel_size
        )
        grid = (
            coords - grid_origin[:, None] + self.config.voxel_size / 2
        ) / crop_size_m * 2 - 1
        point_valid = (
            torch.nn.functional.grid_sample(
                voxel_valid[:, None].float(),
                grid[:, None, None, :, [2, 1, 0]],
                align_corners=False,
                mode="nearest",
                padding_mode="zeros",
            )[:, 0, 0, 0]
            > 0.5
        )

        point_feats = torch.nn.functional.grid_sample(
            voxel_feats,
            grid[:, None, None, :, [2, 1, 0]],
            align_corners=False,
            mode="bilinear",
            padding_mode="zeros",
        )[:, :, 0, 0]
        return point_feats, point_valid

    def predict_init(self, batch, session: ReconSession) -> None:
        # setup before starting inference on a new scan
        torch.cuda.synchronize()
        session.log.init_time_0 = time.time()

        vox4 = self.config.voxel_size * 4
        minbound = batch["gt_origin"][0]
        maxbound = batch["gt_maxbound"][0].float()
        maxbound = (torch.ceil((maxbound - minbound) / vox4) - 0.001) * vox4 + minbound

        x = torch.arange(
            minbound[0], maxbound[0], self.config.voxel_size, dtype=torch.float32
        )
        y = torch.arange(
            minbound[1], maxbound[1], self.config.voxel_size, dtype=torch.float32
        )
        z = torch.arange(
            minbound[2], maxbound[2], self.config.voxel_size, dtype=torch.float32
        )
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        session.infer.global_coords = torch.stack((xx, yy, zz), dim=-1).to(self.config.device)

        nvox = xx.shape
        session.infer.running_count = torch.zeros(nvox, dtype=torch.float32, device=self.config.device)
        session.infer.M = torch.zeros(
            (self.fusion.out_c, *nvox),
            dtype=torch.float32,
            device=self.config.device,
        )

        if self.depth_guidance.enabled:
            session.infer.depths = []

            if self.depth_guidance.density_fusion_channel:
                session.infer.running_density = torch.zeros(
                    nvox, dtype=torch.float32, device=self.config.device
                )
                session.infer.running_density_weight = torch.zeros(
                    nvox, dtype=torch.int32, device=self.config.device
                )
            elif self.depth_guidance.tsdf_fusion_channel:
                session.infer.running_tsdf = torch.zeros(
                    nvox, dtype=torch.float32, device=self.config.device
                )
                session.infer.running_tsdf_weight = torch.zeros(
                    nvox, dtype=torch.int32, device=self.config.device
                )

        torch.cuda.synchronize()
        session.log.init_time_1 = time.time()
        session.log.n_inits += 1
    
    def predict_per_view(self, batch, session: ReconSession) -> None:
        # fuse each view into the scene volume
        t0 = time.time()
        torch.cuda.synchronize()

        batch_size, n_imgs, _, imheight, imwidth = batch["images"].shape
        imsize = imheight, imwidth
        assert batch_size == 1 and n_imgs == 1

        uv, z, valid = utils.project(
            session.infer.global_coords[None],
            batch["poses"][None],
            batch["k_image"][None],
            imsize,
        )
        valid = valid[0, 0]
        coords = session.infer.global_coords[valid][None, None, None]

        if self.depth_guidance.enabled:
            (
                img_voxel_feats,
                img_voxel_valid,
            ) = self.get_img_voxel_feats_by_depth_guided_bp(
                batch["images"],
                batch["depths"],
                batch["poses"][None],
                batch["k_image"][None],
                batch["k_depth"][None],
                coords,
            )
            if self.depth_guidance.density_fusion_channel:
                density, density_weight = utils.density_fusion(
                    batch["depths"],
                    batch["poses"][None],
                    batch["k_depth"][:, None],
                    coords,
                    self.config.voxel_size
                )
                density = density[0, 0, 0]
                density_weight = density_weight[0, 0, 0]
            elif self.depth_guidance.tsdf_fusion_channel:
                tsdf, tsdf_weight = utils.tsdf_fusion(
                    batch["depths"],
                    batch["poses"][None],
                    batch["k_depth"][:, None],
                    coords,
                    self.config.voxel_size
                )
                tsdf = tsdf[0, 0, 0]
                tsdf_weight = tsdf_weight[0, 0, 0]
                tsdf.masked_fill_(tsdf_weight == 0, 0)
        else:
            (img_voxel_feats, img_voxel_valid,) = self.get_img_voxel_feats_by_img_bp(
                batch["images"],
                batch["poses"][None],
                batch["k_image"][None],
                coords,
            )

        """
        in get_img_voxel_feats_by_img_bp these values are already zeroed inside of utils.sample_voxel_feats
        zeroing again here just in case
        """
        img_voxel_feats.masked_fill_(~img_voxel_valid[:, :, None], 0)

        old_count = session.infer.running_count[valid].clone()
        session.infer.running_count[valid] += img_voxel_valid[0, 0, 0, 0]
        new_count = session.infer.running_count[valid]

        x = img_voxel_feats[0, 0, :, 0, 0]
        old_m = session.infer.M[:, valid]
        new_m = x / new_count[None] + (old_count / new_count)[None] * old_m
        session.infer.M[:, valid] = new_m
        session.infer.M.masked_fill_(session.infer.running_count[None] == 0, 0)

        if self.depth_guidance.enabled:
            if self.depth_guidance.density_fusion_channel:
                old_count = session.infer.running_density_weight[valid]
                session.infer.running_density_weight[valid] += density_weight
                new_count = session.infer.running_density_weight[valid]
                denom = new_count + (new_count == 0)
                session.infer.running_density[valid] = (
                    density / denom + (old_count / denom) * session.infer.running_density[valid]
                )
            elif self.depth_guidance.tsdf_fusion_channel:
                old_count = session.infer.running_tsdf_weight[valid]
                session.infer.running_tsdf_weight[valid] += tsdf_weight
                new_count = session.infer.running_tsdf_weight[valid]
                denom = new_count + (new_count == 0)
                session.infer.running_tsdf[valid] = (
                    tsdf / denom + (old_count / denom) * session.infer.running_tsdf[valid]
                )

        t1 = time.time()
        torch.cuda.synchronize()
        session.log.per_view_time += t1 - t0
        session.log.n_views += 1

    def predict_final(
        self, 
        batch: Dict[str, torch.Tensor],
        session: ReconSession,
        device: torch.device,
    ) -> Optional[dict | Any]:
        torch.cuda.synchronize()
        session.log.final_step_time_0 = time.time()

        global_feats = session.infer.M
        global_feats = self.fusion.bn(global_feats[None]).squeeze(0)

        if self.config.no_image_features:
            global_feats = global_feats * 0

        if self.depth_guidance.enabled:
            if self.depth_guidance.density_fusion_channel:
                global_feats = torch.cat(
                    (global_feats, session.infer.running_density[None]), dim=0
                )
            elif self.depth_guidance.tsdf_fusion_channel:
                session.infer.running_tsdf.masked_fill_(session.infer.running_tsdf_weight == 0, 1)

                extra = session.infer.running_tsdf[None]
                global_feats = torch.cat((global_feats, extra), dim=0)

        global_feats = self.cnn3d(global_feats[None], session.infer.running_count[None] > 0)
        global_valid = session.infer.running_count > 0

        coarse_spatial_dims = np.array(global_feats.shape[2:])
        fine_spatial_dims = coarse_spatial_dims * self.config.output_sample_rate

        coarse_occ_logits = self.occ_predictor(
            global_feats.view(1, global_feats.shape[1], -1)
        ).view(global_feats.shape[2:])

        coarse_occ_mask = coarse_occ_logits > 0
        coarse_occ_idx = torch.argwhere(coarse_occ_mask)
        n_coarse_vox_occ = len(coarse_occ_idx)

        fine_surface = torch.full(
            tuple(fine_spatial_dims), torch.nan, device="cpu", dtype=torch.float32
        )

        coarse_voxel_size = self.config.voxel_size
        fine_voxel_size = self.config.voxel_size / self.config.output_sample_rate

        x = torch.arange(self.config.output_sample_rate)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing="ij")
        fine_idx_offset = torch.stack((xx, yy, zz), dim=-1).view(-1, 3).to(self.config.device)
        fine_offset = (
            fine_idx_offset * fine_voxel_size
            - coarse_voxel_size / 2
            + fine_voxel_size / 2
        )

        coarse_voxel_chunk_size = (2**20) // (self.config.output_sample_rate**3)

        if self.config.point_backprojection:
            imheight, imwidth = session.infer.images[0].shape[1:]
            featheight = imheight // 4
            featwidth = imwidth // 4

            keyframe_chunk_size = 32
            highres_img_feats = torch.full(
                (
                    len(session.infer.images),
                    self.cnn2d_pb_out_dim,
                    featheight,
                    featwidth,
                ),
                torch.nan,
                dtype=torch.float32,
                device="cpu",
            )

            for keyframe_chunk_start in tqdm.trange(
                0,
                len(session.infer.images),
                keyframe_chunk_size,
                desc="Highres image features",
                leave=False,
            ):
                keyframe_chunk_end = min(
                    keyframe_chunk_start + keyframe_chunk_size,
                    len(session.infer.images),
                )

                rgb_imgs = torch.stack(
                    session.infer.images[keyframe_chunk_start:keyframe_chunk_end],
                    dim=0,
                ).to(device=device)

                highres_img_feats[
                    keyframe_chunk_start:keyframe_chunk_end
                ] = self.cnn2d_pb(rgb_imgs)

                rgb_imgs = rgb_imgs.to(device="cpu")

        for coarse_voxel_chunk_start in tqdm.trange(
            0, n_coarse_vox_occ, coarse_voxel_chunk_size, leave=False, desc="Chunks"
        ):
            coarse_voxel_chunk_end = min(
                coarse_voxel_chunk_start + coarse_voxel_chunk_size, n_coarse_vox_occ
            )

            chunk_coarse_idx = coarse_occ_idx[
                coarse_voxel_chunk_start:coarse_voxel_chunk_end
            ]
            chunk_coarse_coords = (
                chunk_coarse_idx * coarse_voxel_size + batch["gt_origin"]
            )

            chunk_fine_coords = chunk_coarse_coords[:, None].repeat(
                1, self.config.output_sample_rate**3, 1
            )
            chunk_fine_coords += fine_offset[None]
            chunk_fine_coords = chunk_fine_coords.view(-1, 3)

            (
                chunk_fine_feats,
                chunk_fine_valid,
            ) = self.sample_point_features_by_linear_interp(
                chunk_fine_coords,
                global_feats,
                global_valid[None],
                batch["gt_origin"],
            )

            if self.config.point_backprojection:
                img_feature_dim = session.infer.M.shape[0]
                fine_bp_feats = torch.zeros(
                    (self.cnn2d_pb_out_dim, len(chunk_fine_coords)),
                    device=self.config.device,
                    dtype=session.infer.M.dtype,
                )
                counts = torch.zeros(
                    len(chunk_fine_coords), device=self.config.device, dtype=torch.float32
                )

                if self.depth_guidance.enabled:
                    if self.depth_guidance.density_fusion_channel:
                        fine_density = torch.zeros(
                            len(chunk_fine_coords), device=self.config.device
                        )
                        fine_density_weights = torch.zeros(
                            len(chunk_fine_coords),
                            device=self.config.device,
                            dtype=torch.float32,
                        )
                    elif self.depth_guidance.tsdf_fusion_channel:
                        fine_tsdf = torch.zeros(
                            len(chunk_fine_coords), device=self.config.device
                        )
                        fine_tsdf_weights = torch.zeros(
                            len(chunk_fine_coords),
                            device=self.config.device,
                            dtype=torch.float32,
                        )

                for keyframe_chunk_start in range(
                    0, len(session.infer.images), keyframe_chunk_size
                ):
                    keyframe_chunk_end = min(
                        keyframe_chunk_start + keyframe_chunk_size,
                        len(session.infer.images),
                    )

                    chunk_highres_img_feats = highres_img_feats[
                        keyframe_chunk_start:keyframe_chunk_end
                    ].to(self.config.device)
                    rgb_img_placeholder = torch.empty(
                        1, len(chunk_highres_img_feats), 3, imheight, imwidth
                    )

                    poses = torch.stack(
                        session.infer.poses[keyframe_chunk_start:keyframe_chunk_end],
                        dim=0,
                    )

                    if self.depth_guidance.enabled:
                        pred_depth_imgs = torch.stack(
                            session.infer.depths[
                                keyframe_chunk_start:keyframe_chunk_end
                            ],
                            dim=0,
                        )
                        (
                            _fine_bp_feats,
                            valid,
                        ) = self.get_img_voxel_feats_by_depth_guided_bp(
                            rgb_img_placeholder,
                            pred_depth_imgs[None],
                            poses[None],
                            batch["k_image"][:, None],
                            batch["k_depth"][:, None],
                            chunk_fine_coords[None],
                            use_highres_cnn=True,
                            img_feats=chunk_highres_img_feats,
                        )
                    else:
                        _fine_bp_feats, valid = self.get_img_voxel_feats_by_img_bp(
                            rgb_img_placeholder,
                            poses[None],
                            batch["k_image"][:, None],
                            chunk_fine_coords[None],
                            use_highres_cnn=True,
                            img_feats=chunk_highres_img_feats,
                        )

                    old_counts = counts.clone()
                    current_counts = valid.squeeze(0).sum(dim=0)
                    counts += current_counts

                    denom = torch.clamp_min(counts, 1)
                    _fine_bp_feats = _fine_bp_feats.squeeze(0)
                    _fine_bp_feats /= denom
                    _fine_bp_feats = _fine_bp_feats.sum(dim=0)
                    fine_bp_feats *= old_counts / denom
                    fine_bp_feats += _fine_bp_feats

                    if self.depth_guidance.enabled:
                        if self.depth_guidance.density_fusion_channel:
                            density, weight = utils.density_fusion(
                                pred_depth_imgs[None],
                                poses[None],
                                batch["k_depth"][:, None],
                                chunk_fine_coords[None],
                                self.config.voxel_size
                            )
                            old_count = fine_density_weights.clone()
                            fine_density_weights += weight.squeeze(0)
                            new_count = fine_density_weights
                            denom = torch.clamp_min(new_count, 1)
                            fine_density = (
                                density.squeeze(0) / denom
                                + (old_count / denom) * fine_density
                            )
                        elif self.depth_guidance.tsdf_fusion_channel:
                            tsdf, weight = utils.tsdf_fusion(
                                pred_depth_imgs[None],
                                poses[None],
                                batch["k_depth"][:, None],
                                chunk_fine_coords[None],
                                self.config.voxel_size
                            )
                            tsdf.masked_fill_(weight == 0, 0)

                            old_count = fine_tsdf_weights.clone()
                            fine_tsdf_weights += weight.squeeze(0)
                            new_count = fine_tsdf_weights
                            denom = torch.clamp_min(new_count, 1)
                            fine_tsdf = (
                                tsdf.squeeze(0) / denom
                                + (old_count / denom) * fine_tsdf
                            )

                fine_bp_feats = self.point_fusion.bn(
                    fine_bp_feats[None, ..., None, None]
                )[..., 0, 0]
                fine_bp_feats = self.point_feat_mlp(fine_bp_feats)

                if self.config.no_image_features:
                    fine_bp_feats = fine_bp_feats * 0

                if self.depth_guidance.enabled:
                    if self.depth_guidance.density_fusion_channel:
                        chunk_fine_feats = torch.cat(
                            (fine_bp_feats, chunk_fine_feats, fine_density[None, None]),
                            dim=1,
                        )
                    elif self.depth_guidance.tsdf_fusion_channel:
                        fine_tsdf.masked_fill_(fine_tsdf_weights == 0, 1)

                        extra = fine_tsdf[None]

                        chunk_fine_feats = torch.cat(
                            (fine_bp_feats, chunk_fine_feats, extra[None]), dim=1
                        )
                    else:
                        chunk_fine_feats = torch.cat(
                            (fine_bp_feats, chunk_fine_feats), dim=1
                        )
                else:
                    chunk_fine_feats = torch.cat(
                        (fine_bp_feats, chunk_fine_feats), dim=1
                    )

            chunk_fine_surface_logits = (
                self.surface_predictor(chunk_fine_feats)[0, 0].cpu().float()
            )

            chunk_fine_idx = chunk_coarse_idx[:, None].repeat(
                1, self.config.output_sample_rate**3, 1
            )
            chunk_fine_idx *= self.config.output_sample_rate
            chunk_fine_idx += fine_idx_offset[None]
            chunk_fine_idx = chunk_fine_idx.view(-1, 3).cpu()

            fine_surface[
                chunk_fine_idx[:, 0],
                chunk_fine_idx[:, 1],
                chunk_fine_idx[:, 2],
            ] = chunk_fine_surface_logits

        torch.tanh_(fine_surface)
        fine_surface *= 0.5
        fine_surface += 0.5

        torch.cuda.synchronize()
        session.log.final_step_time_1 = time.time()
        session.log.n_final_steps += 1

        out_path = os.getcwd() if session.out_path is None else session.out_path
        log_path = os.path.join(out_path, f"{session.user_id}/recon")
        name = f"{time.time():.4f}_" + session.task_id + "_recon"
        os.makedirs(log_path, exist_ok=True)

        origin = (
            batch["gt_origin"].cpu().numpy()[0]
            - coarse_voxel_size / 2
            + fine_voxel_size / 2
        )

        try:
            pred_mesh = utils.tsdf2mesh(
                fine_surface.numpy(),
                voxel_size=fine_voxel_size,
                origin=origin,
                level=0.5,
            )

            # K = session.infer.k_color[0]
            # V = pred_mesh.vertices.shape[0]
            # verts_world = np.hstack([
            #     pred_mesh.vertices.astype(np.float32),
            #     np.ones((V, 1), dtype=np.float32)
            # ])

            # color_accum = np.zeros((V, 3), dtype=np.float64)
            # weight_accum = np.zeros((V,), dtype=np.float64)
            
            # for image_idx in range(len(session.infer.images)):
            #     T_cw = poses[image_idx]
            #     img = session.infer.images[image_idx]

            #     verts_cam = (T_cw @ verts_world.T).T
            #     xc = verts_cam[:, 0]
            #     zc = verts_cam[:, 2]
            #     yc = verts_cam[:, 1]

            #     valid = zc > 0
            #     if not np.any(valid):
            #         continue

            #     idx_valid = np.where(valid)[0]

            #     x_valid = xc[idx_valid]
            #     y_valid = yc[idx_valid]
            #     z_valid = zc[idx_valid]

            #     x_norm = x_valid / z_valid
            #     y_norm = y_valid / z_valid

            #     uv_h = (K @ np.vstack([x_norm, y_norm, np.ones_like(x_norm)])).T  # (N_valid, 3)
            #     u = uv_h[:, 0] / uv_h[:, 2]
            #     v = uv_h[:, 1] / uv_h[:, 2]

            #     in_img = (u >= 0) & (u < imwidth - 1) & (v >= 0) & (v < imheight - 1)
            #     if not np.any(in_img):
            #         continue
                
            #     idx_final = idx_valid[in_img]
            #     u_sel = u[in_img]
            #     v_sel = v[in_img]

            #     u_int = np.round(u_sel).astype(int)
            #     v_int = np.round(v_sel).astype(int)

            #     sampled_colors = img[v_int, u_int, :].astype(np.float64) / 255.0  # [0,1]
            #     w = np.ones_like(u_int, dtype=np.float64)  # 지금은 uniform weight

            #     color_accum[idx_final] += sampled_colors * w[:, None]
            #     weight_accum[idx_final] += w

            # default_color = np.asarray(default_color, dtype=np.float32)
            # if default_color.shape != (3,):
            #     raise ValueError("default_color must be a sequence of length 3 (r,g,b).")

            # vertex_colors = np.zeros((V, 3), dtype=np.float32)
            # valid_color = weight_accum > 0

            # vertex_colors[valid_color] = (color_accum[valid_color] /
            #                               weight_accum[valid_color, None]).astype(np.float32)

            # vertex_colors[~valid_color] = default_color

            # pred_mesh.visual.vertex_colors = (vertex_colors * 255.0).clip(0, 255).astype(np.uint8)

        except Exception as e:
            print(e)
        else:
            pred_mesh.export(
                os.path.join(log_path, f"{name}.glb"), 
                file_type="glb"
            )

            return pred_mesh.export(file_type="dict")

    # args, kwargs: unreferenced parameters
    def on_predict_epoch_end(self, session: ReconSession):
        init_time = session.log.init_time_1 - session.log.init_time_0

        per_init_time = init_time / session.log.n_inits
        per_view_time = session.log.per_view_time / session.log.n_views
        final_step_time = (session.log.final_step_time_1 - session.log.final_step_time_0) / session.log.n_final_steps
        logging.info(f"{session.task_id} - per_init_time: {per_init_time:.4f}")
        logging.info(f"{session.task_id} - per_view_time: {per_view_time:.4f}")
        logging.info(f"{session.task_id} - final_step_time: {final_step_time:.4f}")


class ReconPredictor:
    def __init__(self, config: box.Box):
        self.config = config
        
        checkpoint_uri = os.path.join(os.getcwd(), config.checkpoints)
        self.predictor = ReconPro(config)
        self.predictor.load_state_dict(
            torch.load(checkpoint_uri, map_location="cpu")
        )
        
        self.predictor.eval()
        
    def init(self):
        self.predictor.to(self.config.device)
    
    def infer(
        self, 
        batch: Dict[str, torch.Tensor], 
        user_id: str,
        task_id: str, 
        log: ReconLog,
        device: torch.device
    ) -> Optional[Dict | Any]:
        # import pdb; pdb.set_trace()
        if (not self._check_batch(batch)):
            raise ValueError("Input batch is missing required keys.")

        # 잘못된 batch를 수정
        batch = self._update_gt_if_invalid(batch)

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        session = ReconSession(
            user_id=user_id,
            task_id=task_id,
            out_path=os.path.join(os.getcwd(), "logs/"),
            infer=ReconInferArray(),
            log=log
        )

        self.predictor.predict_init(batch, session=session)

        batch_iterator = data.ReconIterator(batch, enable_padding=True)
        batch_step: int = 1
        batch_length = batch_iterator.length // batch_step
        glb_bytes: Optional[Dict | Any] = None

        # import pdb; pdb.set_trace()
        for frame_idx, frame in tqdm.tqdm(
            enumerate(batch_iterator), 
            total=batch_length, 
            desc="Extract image features"
        ):
            frame = self._transfer_batch_to_device(frame, device)
            self.predictor.predict_per_view(frame, session=session)

            if self.config.point_backprojection:
                # store any frames that are marked as keyframes for later point back-projection
                session.infer.images.append(frame["images"][0, 0].cpu())
                session.infer.poses.append(frame["poses"][0])
                if self.predictor.depth_guidance.enabled:
                    session.infer.depths.append(frame["depths"][0, 0])

                session.infer.k_color.append(frame["k_image"].cpu())
                session.infer.k_depth.append(frame["k_depth"].cpu())

            if (frame_idx == (batch_length) - 1):
                # import pdb; pdb.set_trace()
                logging.info(f"Start inference final step. task id: {task_id}")
                glb_bytes = self.predictor.predict_final(
                    frame, 
                    session=session,
                    device=device,
                )

                self.predictor.on_predict_epoch_end(
                    session=session
                )

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        return glb_bytes

    def _check_batch(self, batch):
        required_keys = [
            "images",
            "depths",
            "poses",
            "k_image",
            "k_depth",
            "gt_origin",
            "gt_maxbound",
        ]

        for k in required_keys:
            if k not in batch:
                return False
        
        for k in required_keys:
            if not len(batch[k]):
                return False
            
        return True
    
    def _update_gt_if_invalid(self, batch: dict) -> Any | None:
        try:
            gt_origin = batch['gt_origin']
            if (gt_origin.shape[0] != 1):
                batch['gt_origin'] = batch['gt_origin'][0, None]
                logging.warning(f"unexpected gt_origin found. change gt_origin shape [{gt_origin.shape[0]}, {gt_origin.shape[1]}] to [1, 3].")
            
            gt_maxbound = batch['gt_maxbound']
            if (gt_maxbound.shape[0] != 1):
                batch['gt_maxbound'] = batch['gt_maxbound'][0, None]
                logging.warning(
                    f"unexpected gt_maxbound found. change gt_maxbound shape [{gt_maxbound.shape[0]}, {gt_maxbound.shape[1]}] to [1, 3]."
                )

        except Exception as e:
            logging.error(f"ERROR: {e}")
            
        else:
            return batch
        
    def _transfer_batch_to_device(
        self, 
        batch: Dict[str, torch.Tensor], 
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        return batch