import os
import sys
import cv2
import time
import torch
import tqdm
import numpy as np
from typing import Tuple

from torchvision.transforms import (
    Lambda,
    Compose,
    ToTensor,
    Normalize,
    ConvertImageDtype,
)

if os.path.join(os.path.abspath(os.curdir)) not in sys.path:
    sys.path.append(os.path.join(os.path.abspath(os.curdir)))

from source.model.modules import (
    Cnn2d, Cnn3d,
    FeatureFusion,
    ResBlock1d,
    MultiresConvDecoder,
    DepthProEncoder,
    FOVNetwork
)

from source.model.utils import (
    sample_posed_images,
    sample_voxel_feats,
    create_backbone_model,
    augment_depth_inplace,
    compute_recon_loss,
    density_fusion,
    tsdf_fusion,
    project,
    tsdf2mesh,

    sample_point_features_by_linear_interp,
    convert_to_color_depth_lut
)

from source.model.loaders.loaders import (
    UniversalToTensor
)

class ArvaneContainer:
    def __init__(self):
        self.color = list()
        self.depth = list()
        self.poses = list()
        self.k_color = list()
        self.k_depth = list()

        self.gt_origin = None
        self.gt_maxbound = None

        self.M = None
        self.running_count = None
        self.running_density = None
        self.running_density_weight = None
        self.running_tsdf = None
        self.running_tsdf_weight = None
        self.init_time = None
        self.n_inits = None

    
class ArvaneModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.depth_config = config.depth_estimation
        self.training = False
        
        self.container = ArvaneContainer()
        self.device = torch.device(self.config.device)

        # --------depth--------
        patch_encoder, patch_encoder_config = create_backbone_model(
            preset=self.depth_config.patch_encoder_preset
        )
        image_encoder, _ = create_backbone_model(
            preset=self.depth_config.image_encoder_preset
        )

        fov_encoder = None
        if self.depth_config.use_fov_head and self.depth_config.fov_encoder_preset is not None:
            fov_encoder, _ = create_backbone_model(preset=self.depth_config.fov_encoder_preset)

        dims_encoder = patch_encoder_config.encoder_feature_dims
        hook_block_ids = patch_encoder_config.encoder_feature_layer_ids

        self.depth_encoder = DepthProEncoder(
            dims_encoder=dims_encoder,
            patch_encoder=patch_encoder,
            image_encoder=image_encoder,
            hook_block_ids=hook_block_ids,
            decoder_features=self.depth_config.decoder_features,
        )
        self.depth_decoder = MultiresConvDecoder(
            dims_encoder=[self.depth_config.decoder_features] + list(self.depth_encoder.dims_encoder),
            dim_decoder=self.depth_config.decoder_features,
        )

        last_dims=(32, 1)
        dim_decoder = self.depth_decoder.dim_decoder
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(
                dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            torch.nn.Conv2d(
                dim_decoder // 2,
                last_dims[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
        )

        # Set the final convolution layer's bias to be 0.
        self.head[4].bias.data.fill_(0)

        # Set the FOV estimation head.
        if self.depth_config.use_fov_head:
            self.fov = FOVNetwork(num_features=dim_decoder, fov_encoder=fov_encoder)

        self.depth_transform = Compose(
            [
                UniversalToTensor(),
                Lambda(lambda x: x.to(self.config.device)),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ConvertImageDtype(torch.float32),
            ]
        )
            
        # --------recon--------
        img_feature_dim = 47
        self.cnn2d = Cnn2d(out_dim=img_feature_dim)
        self.fusion = FeatureFusion(in_c=img_feature_dim)
        self.voxel_feat_dim = self.fusion.out_c

        self.depth_guidance = self.config.depth_guidance

        if self.depth_guidance.enabled:
            if self.depth_guidance.density_fusion_channel:
                self.voxel_feat_dim += 1
            elif self.depth_guidance.tsdf_fusion_channel:
                self.voxel_feat_dim += 1

        self.cnn3d = Cnn3d(in_c=self.voxel_feat_dim)

        if self.config.point_backprojection:
            self.cnn2d_pb_out_dim = img_feature_dim
            self.cnn2d_pb = Cnn2d(out_dim=self.cnn2d_pb_out_dim)
            self.point_feat_mlp = torch.nn.Sequential(
                ResBlock1d(self.cnn2d_pb_out_dim),
                ResBlock1d(self.cnn2d_pb_out_dim),
            )
            self.point_fusion = FeatureFusion(in_c=self.cnn2d_pb_out_dim)

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
            ResBlock1d(32),
            ResBlock1d(32),
            torch.nn.Conv1d(32, 1, 1),
        )
        self.occ_predictor = torch.nn.Sequential(
            torch.nn.Conv1d(occ_pred_input_dim, 32, 1),
            ResBlock1d(32),
            ResBlock1d(32),
            torch.nn.Conv1d(32, 1, 1),
        )

        if self.config.do_prediction_timing:
            self.init_time = 0
            self.per_view_time = 0
            self.final_step_time = 0
            self.n_final_steps = 0
            self.n_views = 0
            self.n_inits = 0

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.001)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda epoch: 1
            if self.global_step < (self.config.steps - self.config.finetune_steps)
            else 0.1,
            verbose=True,
        )
        return [opt], [sched]
    
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

        depth, depth_valid, z = sample_posed_images(
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

        img_voxel_feats, img_voxel_valid = sample_voxel_feats(
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
            with torch.autocast(enabled=False, device_type=self.device.type):
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

    def transfer_batch_to_device(self, batch):
        self.transfer_keys = [
            "input_coords",
            "output_coords",
            "crop_center",
            "crop_rotation",
            "crop_size_m",
            "gt_tsdf",
            "gt_occ",
            "K_color",
            "K_depth",
            "color",
            "depth",
            "poses",
            "gt_origin",
            "gt_maxbound",
        ]
        self.no_transfer_keys = [
            "scan_name",
            "gt_tsdf_npzfile",
            "keyframe",
            "initial_frame",
            "final_frame",
        ]

        transfer_batch = {}
        no_transfer_batch = {}
        for k in batch:
            if k in self.transfer_keys:
                transfer_batch[k] = batch[k]
            elif k in self.no_transfer_keys:
                no_transfer_batch[k] = batch[k]
            else:
                raise NotImplementedError

        for key, value in transfer_batch.items():
            transfer_batch[key] = value.to(self.config.device)

        return transfer_batch
    
    @property
    def img_size(self) -> int:
        """Return the internal image size of the network."""
        return self.depth_encoder.img_size
    
    def forward(self, batch) -> torch.Tensor:
        # --------depth--------
        color = batch['color'] # 정규화 해제 후 원본 이미지 복원
        B, T, H, W, C = color.shape
        resize = H != self.img_size or W != self.img_size
        depths = torch.empty((0, H, W),
                         dtype=torch.float32,
                         device=self.config.device)
        
        flatten_color = self.depth_transform(
            color.squeeze()
        )

        if resize:
            flatten_color = torch.nn.functional.interpolate(
                flatten_color.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )

        encodings = self.depth_encoder(flatten_color)
        features, features_0 = self.depth_decoder(encodings)
        canonical_inverse_depth = self.head(features)

        fov_deg = None
        if hasattr(self, "fov"):
            fov_deg = self.fov.forward(flatten_color, features_0.detach())
            f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))

        inverse_depth = canonical_inverse_depth * (W / f_px)
        f_px = f_px.squeeze()

        if resize:
            inverse_depth = torch.nn.functional.interpolate(
                inverse_depth, size=(H, W), mode="bilinear", align_corners=False
            )

        depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4).squeeze(0)
        depths = torch.cat(
            [depths, depth],
            dim=0
        )

        batch['depth'] = depths

        # --------recon--------
        voxel_feats, voxel_valid = self.get_img_voxel_feats_by_depth_guided_bp(
            batch["color"],
            batch["depth"],
            batch["poses"],
            batch["K_color"][:, None],
            batch["K_depth"][:, None],
            batch["input_coords"],
        )
        voxel_feats = self.fusion(voxel_feats, voxel_valid)
        voxel_valid = voxel_valid.sum(dim=1) > 1
        if self.config.no_image_features:
            voxel_feats = voxel_feats * 0

            if self.depth_guidance.density_fusion_channel:
                density, weight = density_fusion(
                    self.config.voxel_size,
                    batch["depth"],
                    batch["poses"],
                    batch["K_depth"][:, None],
                    batch["input_coords"],
                )
                voxel_feats = torch.cat((voxel_feats, density[:, None]), dim=1)
            elif self.depth_guidance.tsdf_fusion_channel:
                tsdf, weight = tsdf_fusion(
                    self.config.voxel_size,
                    batch["depth"],
                    batch["poses"],
                    batch["K_depth"][:, None],
                    batch["input_coords"],
                )
                tsdf.masked_fill_(weight == 0, 1)
                voxel_feats = torch.cat((voxel_feats, tsdf[:, None]), dim=1)
        else:
            voxel_feats, voxel_valid = self.get_img_voxel_feats_by_img_bp(
                batch["color"],
                batch["poses"],
                batch["K_color"][:, None],
                batch["input_coords"],
            )
            voxel_feats = self.fusion(voxel_feats, voxel_valid)
            voxel_valid = voxel_valid.sum(dim=1) > 1
        
        voxel_feats = self.cnn3d(voxel_feats, voxel_valid)

        if self.config.improved_tsdf_sampling:
            """
            interpolate the features to the points where we have GT tsdf
            """

            t = batch["crop_center"]
            R = batch["crop_rotation"]
            coords = batch["output_coords"]

            with torch.autocast(enabled=False, device_type=self.device.type):
                coords_local = (coords - t[:, None]) @ R
            coords_local += batch["crop_size_m"][:, None] / 2
            origin = torch.zeros_like(batch["gt_origin"])

            (
                coarse_point_feats,
                coarse_point_valid,
            ) = self.sample_point_features_by_linear_interp(
                coords_local, voxel_feats, voxel_valid, origin
            )
        else:
            """
            keep the voxel-center features that we have:
            GT tsdf has already been interpolated to these points
            """

            coarse_point_feats = voxel_feats.view(*voxel_feats.shape[:2], -1)
            coarse_point_valid = voxel_valid.view(voxel_valid.shape[0], -1)

        if self.config.point_backprojection:
            coords = batch["output_coords"]

            if self.depth_guidance.enabled:
                (
                    fine_point_feats,
                    fine_point_valid,
                ) = self.get_img_voxel_feats_by_depth_guided_bp(
                    batch["color"],
                    batch["depth"],
                    batch["poses"],
                    batch["K_color"][:, None],
                    batch["K_depth"][:, None],
                    coords,
                    use_highres_cnn=True,
                )
            else:
                (
                    fine_point_feats,
                    fine_point_valid,
                ) = self.get_img_voxel_feats_by_img_bp(
                    batch["color"],
                    batch["poses"],
                    batch["K_color"][:, None],
                    coords,
                    use_highres_cnn=True,
                )

            fine_point_feats = self.point_fusion(
                fine_point_feats[..., None, None], fine_point_valid[..., None, None]
            )[..., 0, 0]
            fine_point_valid = coarse_point_valid & (fine_point_valid.any(dim=1))
            fine_point_feats = self.point_feat_mlp(fine_point_feats)

            if self.config.no_image_features:
                fine_point_feats = fine_point_feats * 0

            if self.depth_guidance.enabled:
                if self.depth_guidance.density_fusion_channel:
                    density, weight = density_fusion(
                        batch["depth"],
                        batch["poses"],
                        batch["K_depth"][:, None],
                        coords,
                    )
                    fine_point_feats = torch.cat(
                        (fine_point_feats, coarse_point_feats, density[:, None]), dim=1
                    )
                elif self.depth_guidance.tsdf_fusion_channel:
                    tsdf, weight = tsdf_fusion(
                        batch["depth"],
                        batch["poses"],
                        batch["K_depth"][:, None],
                        coords,
                    )
                    tsdf.masked_fill_(weight == 0, 1)

                    fine_point_feats = torch.cat(
                        (fine_point_feats, coarse_point_feats, tsdf[:, None]), dim=1
                    )
                else:
                    fine_point_feats = torch.cat(
                        (fine_point_feats, coarse_point_feats), dim=1
                    )
            else:
                fine_point_feats = torch.cat(
                    (fine_point_feats, coarse_point_feats), dim=1
                )
        else:
            fine_point_feats = coarse_point_feats
            fine_point_valid = coarse_point_valid

        tsdf_logits = self.surface_predictor(fine_point_feats).squeeze(1)
        occ_logits = self.occ_predictor(coarse_point_feats).squeeze(1)

        # loss, tsdf_loss, occ_loss = compute_recon_loss(
        #     tsdf_logits,
        #     occ_logits,
        #     batch["gt_tsdf"],
        #     batch["gt_occ"],
        #     coarse_point_valid,
        #     fine_point_valid,
        # )

        outputs = {
            # "loss": loss,
            # "tsdf_loss": tsdf_loss,
            # "occ_loss": occ_loss,
            "tsdf_logits": tsdf_logits,
            "occ_logits": occ_logits,
        }

        return outputs
    
    def start(
        self, 
        gt_origin,
        gt_maxbound
    ):
        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t0 = time.time()

        self.container.gt_origin = gt_origin.to(self.config.device)
        self.container.gt_maxbound = gt_maxbound.to(self.config.device)

        vox4 = self.config.voxel_size * 4
        minbound = gt_origin[0]
        maxbound = gt_maxbound[0].float()
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
        self.global_coords = torch.stack((xx, yy, zz), dim=-1).to(self.config.device)

        nvox = xx.shape
        self.container.running_count = torch.zeros(nvox, dtype=torch.float32, device=self.config.device)
        self.container.M = torch.zeros(
            (self.fusion.out_c, *nvox),
            dtype=torch.float32,
            device=self.config.device,
        )

        self.container.keyframe_rgb = []
        self.container.keyframe_pose = []

        if self.depth_guidance.enabled:
            self.container.keyframe_depth = []

            if self.depth_guidance.density_fusion_channel:
                self.container.running_density = torch.zeros(
                    nvox, dtype=torch.float32, device=self.config.device
                )
                self.container.running_density_weight = torch.zeros(
                    nvox, dtype=torch.int32, device=self.config.device
                )
            elif self.depth_guidance.tsdf_fusion_channel:
                self.container.running_tsdf = torch.zeros(
                    nvox, dtype=torch.float32, device=self.config.device
                )
                self.container.running_tsdf_weight = torch.zeros(
                    nvox, dtype=torch.int32, device=self.config.device
                )

        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t1 = time.time()
            self.init_time += t1 - t0
            self.n_inits += 1
    
    @torch.compile(dynamic=False, fullgraph=True, mode="default")
    def depth_forward(
        self, 
        flatten_color,
    ) -> Tuple[torch.Tensor, torch.Tensor]:    
        # -------------------- depth --------------------
        encodings = self.depth_encoder(flatten_color)
        features, features_0 = self.depth_decoder(encodings)
        canonical_inverse_depth = self.head(features)
    
        # -------------------- focal length  --------------------
        fov_deg = self.fov(flatten_color, features_0.detach())
        return canonical_inverse_depth, fov_deg
    
    @torch.compile(dynamic=True, fullgraph=True, mode="default")
    def density_fusion_forward(
        self,
        depth,
        poses,
        k_depth,
        coords,
        voxel_size
    ):
        density, density_weight = density_fusion(
            voxel_size,
            depth,
            poses,
            k_depth,
            coords,
        )
        density = density[0, 0, 0]
        density_weight = density_weight[0, 0, 0]
        return density, density_weight
    
    @torch.compile(dynamic=True, fullgraph=True, mode="default")
    def tsdf_fusion_forward(
        self,
        depth,
        poses,
        k_depth,
        coords,
        voxel_size
    ):
        tsdf, tsdf_weight = tsdf_fusion(
            voxel_size,
            depth,
            poses,
            k_depth,
            coords,
        )
        tsdf = tsdf[0, 0, 0]
        tsdf_weight = tsdf_weight[0, 0, 0]
        tsdf.masked_fill_(tsdf_weight == 0, 0)
        return tsdf, tsdf_weight

    @torch.no_grad()
    def depth_infer(
        self, 
        flatten_color,
        imwidth,
        imheight,
        interpolation_mode="bilinear",
        resize=True,
    ):
        if resize:
            flatten_color = torch.nn.functional.interpolate(
                flatten_color,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )

        canonical_inverse_depth, fov_deg = self.depth_forward(
            flatten_color
        )

        f_px = 0.5 * imwidth / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
        
        inverse_depth = canonical_inverse_depth * (imwidth / f_px)
        f_px = f_px.squeeze()

        if resize:
            inverse_depth = torch.nn.functional.interpolate(
                inverse_depth, size=(imheight, imwidth), mode=interpolation_mode, align_corners=False
            )
        
        depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
        return depth, f_px
    
    @torch.no_grad()
    def create_depth_before_update(
        self,
        color,
        save_to_file=False,
        save_path=None,
        exist_ok=True,
    ):
        current_output_path = save_path if save_path is not None else os.path.join(os.curdir, "depth_temp_outputs")
        if (not save_to_file) and (save_path is not None):
            raise ValueError("If not saving to file, save_path must be None.")
        
        batch_size, n_imgs, _, imheight, imwidth = color.shape
        resize = imheight != self.img_size or imwidth != self.img_size
        assert batch_size == 1 and n_imgs == 1 or resize

        ### make output directory
        os.makedirs(current_output_path, exist_ok=exist_ok)
        os.makedirs(os.path.join(current_output_path, "color"), exist_ok=exist_ok)
        os.makedirs(os.path.join(current_output_path, "depth"), exist_ok=exist_ok)
        os.makedirs(os.path.join(current_output_path, "focal length"), exist_ok=exist_ok)

        result_depth = torch.empty((0, 1, imheight, imwidth), device=self.config.device, dtype=torch.float32)
        result_f_px = torch.empty((0, 1), device=self.config.device, dtype=torch.float32)
        for idx, local_color in enumerate(
            tqdm.tqdm(color, desc="depth estimation", leave=False)
        ):
            # flatten_color: expected shape (1, 3, self.img_size, self.img_size)
            flatten_color = self.depth_transform(local_color)
            
            # depth: expected shape (1, 1, self.img_size, self.img_size)
            # f_px: expected shape (1, ) or None
            depth, f_px = self.depth_infer(
                torch.as_tensor(flatten_color, device=self.config.device, dtype=torch.float32),
                imwidth, imheight
            )

            # result_depth: expected shape (N, 1, self.img_size, self.img_size)
            result_depth = torch.cat(
                [result_depth, depth], dim=0
            )

            # result_f_px: expected shape (N, 1) or (N, None)
            result_f_px = torch.cat(
                [result_f_px, f_px[None, None]], dim=0
            )

            # save depth to file
            if save_to_file:
                # depth_lut: expected shape (self.img_size, self.img_size, 3)
                depth_lut = convert_to_color_depth_lut(
                    depth, dtype=torch.uint8
                ).cpu().numpy().squeeze()

                ### save depth
                cv2.imwrite(
                    os.path.join(current_output_path, "color", f"{idx:04d}.png"), 
                    depth_lut
                )

                ### depth as numpy
                np.save(
                    os.path.join(current_output_path, 'depth', f"{idx:04d}.npy"), 
                    depth[0].cpu().numpy()
                )

                ### f_px as numpy
                if f_px is not None:
                    np.save(
                        os.path.join(current_output_path, 'focal length', f"{idx:04d}.npy"), 
                        f_px.cpu().numpy()
                    )

        # self.container.depth.extend(list(result_depth))
        return result_depth, result_f_px

    def update(
        self, color, depth, k_color, k_depth, poses
    ) -> bool:
        self.container.color  .append(color)
        self.container.depth  .append(depth)
        self.container.k_color.append(k_color)
        self.container.k_depth.append(k_depth)
        self.container.poses  .append(poses)

    def update_view(
        self
    ) -> torch.Tensor:
        color = self.container.color[-1]
        depth = self.container.depth[-1]
        k_color = self.container.k_color[-1]
        k_depth = self.container.k_depth[-1]
        poses = self.container.poses[-1]

        batch_size, n_imgs, _, imheight, imwidth = color.shape
        resize = imheight != self.img_size or imwidth != self.img_size
        assert batch_size == 1 and n_imgs == 1 or resize

        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t0 = time.time()

        # fuse each view into the scene volume
        uv, z, valid = project(
            self.global_coords[None],
            poses[None],
            k_color[None],
            imsize=(imheight, imwidth),
        )
        valid = valid[0, 0]
        coords = self.global_coords[valid][None, None, None]

        if self.depth_guidance.enabled:
            (
                img_voxel_feats,
                img_voxel_valid,
            ) = self.get_img_voxel_feats_by_depth_guided_bp(
                color,
                depth,
                poses[None],
                k_color[None],
                k_depth[None],
                coords,
            )
            if self.depth_guidance.density_fusion_channel:
                density, density_weight = density_fusion(
                    self.config.voxel_size,
                    depth,
                    poses[None],
                    k_depth[None],
                    coords,
                )
                density = density[0, 0, 0]
                density_weight = density_weight[0, 0, 0]
            elif self.depth_guidance.tsdf_fusion_channel:
                tsdf, tsdf_weight = tsdf_fusion(
                    self.config.voxel_size,
                    depth,
                    poses[None],
                    k_depth[None],
                    coords,
                )
                tsdf = tsdf[0, 0, 0]
                tsdf_weight = tsdf_weight[0, 0, 0]
                tsdf.masked_fill_(tsdf_weight == 0, 0)
        else:
            (img_voxel_feats, img_voxel_valid,) = self.get_img_voxel_feats_by_img_bp(
                color,
                poses[None],
                k_color[None],
                coords,
            )

        """
        in get_img_voxel_feats_by_img_bp these values are already zeroed inside of utils.sample_voxel_feats
        zeroing again here just in case
        """
        img_voxel_feats.masked_fill_(~img_voxel_valid[:, :, None], 0)

        old_count = self.container.running_count[valid].clone()
        self.container.running_count[valid] += img_voxel_valid[0, 0, 0, 0]
        new_count = self.container.running_count[valid]

        x = img_voxel_feats[0, 0, :, 0, 0]
        old_m = self.container.M[:, valid]
        new_m = x / new_count[None] + (old_count / new_count)[None] * old_m
        self.container.M[:, valid] = new_m
        self.container.M.masked_fill_(self.container.running_count[None] == 0, 0)

        if self.depth_guidance.enabled:
            if self.depth_guidance.density_fusion_channel:
                old_count = self.container.running_density_weight[valid]
                self.container.running_density_weight[valid] += density_weight
                new_count = self.container.running_density_weight[valid]
                denom = new_count + (new_count == 0)
                self.container.running_density[valid] = (
                    density / denom + (old_count / denom) * self.container.running_density[valid]
                )
            elif self.depth_guidance.tsdf_fusion_channel:
                old_count = self.container.running_tsdf_weight[valid]
                self.container.running_tsdf_weight[valid] += tsdf_weight
                new_count = self.container.running_tsdf_weight[valid]
                denom = new_count + (new_count == 0)
                self.container.running_tsdf[valid] = (
                    tsdf / denom + (old_count / denom) * self.container.running_tsdf[valid]
                )

        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t1 = time.time()
            self.per_view_time += t1 - t0
            self.n_views += 1

    def preview_update(
        self,
        output_path=None,
    ):
        pass

    def final_update(
        self, 
        output_path=None,
    ) -> torch.Tensor:
        color, depth, local_poses, local_k_color, local_k_depth, gt_origin, gt_maxbound = (
            [color[0, 0, ...] for color in self.container.color],
            [depth[0, 0, ...] for depth in self.container.depth],
            [pose[0, ...] for pose in self.container.poses],
            self.container.k_color,
            self.container.k_depth,
            self.container.gt_origin,
            self.container.gt_maxbound,
        )

        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t0 = time.time()

        global_feats = self.container.M
        global_feats = self.fusion.bn(global_feats[None]).squeeze(0)

        if self.config.no_image_features:
            global_feats = global_feats * 0

        if self.depth_guidance.enabled:
            if self.depth_guidance.density_fusion_channel:
                global_feats = torch.cat(
                    (global_feats, self.container.running_density[None]), dim=0
                )
            elif self.depth_guidance.tsdf_fusion_channel:
                self.container.running_tsdf.masked_fill_(self.container.running_tsdf_weight == 0, 1)

                extra = self.container.running_tsdf[None]
                global_feats = torch.cat((global_feats, extra), dim=0)

        global_feats = self.cnn3d(global_feats[None], self.container.running_count[None] > 0)
        global_valid = self.container.running_count > 0

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
            imheight, imwidth = color[0].shape[-2:]
            featheight = imheight // 4
            featwidth = imwidth // 4

            keyframe_chunk_size = 32
            highres_img_feats = torch.full(
                (
                    len(color),
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
                len(color),
                keyframe_chunk_size,
                desc="highres img feats",
                leave=False,
            ):
                keyframe_chunk_end = min(
                    keyframe_chunk_start + keyframe_chunk_size,
                    len(color),
                )

                rgb_imgs = torch.stack(
                    color[keyframe_chunk_start:keyframe_chunk_end],
                    dim=0,
                )

                highres_img_feats[
                    keyframe_chunk_start:keyframe_chunk_end
                ] = self.cnn2d_pb(rgb_imgs)

        for coarse_voxel_chunk_start in tqdm.trange(
            0, n_coarse_vox_occ, coarse_voxel_chunk_size, leave=False, desc="chunks"
        ):
            coarse_voxel_chunk_end = min(
                coarse_voxel_chunk_start + coarse_voxel_chunk_size, n_coarse_vox_occ
            )

            chunk_coarse_idx = coarse_occ_idx[
                coarse_voxel_chunk_start:coarse_voxel_chunk_end
            ]
            chunk_coarse_coords = (
                chunk_coarse_idx * coarse_voxel_size + gt_origin
            )

            chunk_fine_coords = chunk_coarse_coords[:, None].repeat(
                1, self.config.output_sample_rate**3, 1
            )
            chunk_fine_coords += fine_offset[None]
            chunk_fine_coords = chunk_fine_coords.view(-1, 3)

            (
                chunk_fine_feats,
                chunk_fine_valid,
            ) = sample_point_features_by_linear_interp(
                self.config.device,
                self.config.voxel_size,
                chunk_fine_coords,
                global_feats,
                global_valid[None],
                gt_origin,
            )

            if self.config.point_backprojection:
                img_feature_dim = self.container.M.shape[0]
                fine_bp_feats = torch.zeros(
                    (self.cnn2d_pb_out_dim, len(chunk_fine_coords)),
                    device=self.config.device,
                    dtype=self.container.M.dtype,
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
                    0, len(color), keyframe_chunk_size
                ):
                    keyframe_chunk_end = min(
                        keyframe_chunk_start + keyframe_chunk_size,
                        len(color),
                    )

                    chunk_highres_img_feats = highres_img_feats[
                        keyframe_chunk_start:keyframe_chunk_end
                    ].to(self.config.device)
                    rgb_img_placeholder = torch.empty(
                        1, len(chunk_highres_img_feats), 3, imheight, imwidth
                    )

                    poses = torch.stack(
                        self.container.poses[keyframe_chunk_start:keyframe_chunk_end],
                        dim=0,
                    )[:, 0, ...]
                    
                    k_color = torch.stack(
                        self.container.k_color[keyframe_chunk_start:keyframe_chunk_end],
                        dim=0,
                    )[:, 0, ...]

                    k_depth = torch.stack(
                        self.container.k_depth[keyframe_chunk_start:keyframe_chunk_end],
                        dim=0,
                    )[:, 0, ...]

                    if self.depth_guidance.enabled:
                        pred_depth_imgs = torch.stack(
                            depth[
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
                            k_color[None],
                            k_depth[None],
                            chunk_fine_coords[None],
                            use_highres_cnn=True,
                            img_feats=chunk_highres_img_feats,
                        )
                    else:
                        _fine_bp_feats, valid = self.get_img_voxel_feats_by_img_bp(
                            rgb_img_placeholder,
                            poses[None],
                            k_color[None],
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
                            density, weight = density_fusion(
                                self.config.voxel_size,
                                pred_depth_imgs[None],
                                poses[None],
                                k_depth[None],
                                chunk_fine_coords[None],
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
                            tsdf, weight = tsdf_fusion(
                                self.config.voxel_size,
                                pred_depth_imgs[None],
                                poses[None],
                                k_depth[None],
                                chunk_fine_coords[None],
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

        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t1 = time.time()
            self.final_step_time += t1 - t0
            self.n_final_steps += 1

        origin = (
            gt_origin.cpu().numpy()[0]
            - coarse_voxel_size / 2
            + fine_voxel_size / 2
        )

        try:
            pred_mesh = tsdf2mesh(
                fine_surface.numpy(),
                voxel_size=fine_voxel_size,
                origin=origin,
                level=0.5,
            )
        except Exception as e:
            print(e)
        else:
            _ = pred_mesh.export(os.path.join(output_path))
        
        return pred_mesh