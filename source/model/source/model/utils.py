import os
import sys
import box
import timm
import yaml
import types
import torch
import trimesh
import logging
import numpy as np
import scipy.spatial
import skimage.measure
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Any

if os.path.join(os.path.abspath(os.curdir)) not in sys.path:
    sys.path.append(os.path.join(os.path.abspath(os.curdir)))

from source.model.tsdf_fusion import *

try:
    from timm.layers import resample_abs_pos_embed
except ImportError as err:
    print("ImportError: {0}".format(err))

from torch.utils.checkpoint import checkpoint
from typing import Tuple, Literal, Dict, Optional, List
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor,
)

import source.model.loaders.loaders as loaders

@dataclass
class ViTConfig:
    """Configuration for ViT."""

    in_chans: int
    embed_dim: int

    img_size: int = 384
    patch_size: int = 16

    # In case we need to rescale the backbone when loading from timm.
    timm_preset: Optional[str] = None
    timm_img_size: int = 384
    timm_patch_size: int = 16

    # The following 2 parameters are only used by DPT.  See dpt_factory.py.
    encoder_feature_layer_ids: List[int] = None
    """The layers in the Beit/ViT used to constructs encoder features for DPT."""
    encoder_feature_dims: List[int] = None
    """The dimension of features of encoder layers from Beit/ViT features for DPT."""

ViTPreset = Literal[
    "dinov2l16_384",
]

LOGGER = logging.getLogger(__name__)
VIT_CONFIG_DICT: Dict[ViTPreset, ViTConfig] = {
    "dinov2l16_384": ViTConfig(
        in_chans=3,
        embed_dim=1024,
        encoder_feature_layer_ids=[5, 11, 17, 23],
        encoder_feature_dims=[256, 512, 1024, 1024],
        img_size=384,
        patch_size=16,
        timm_preset="vit_large_patch14_dinov2",
        timm_img_size=518,
        timm_patch_size=14,
    ),
}


def make_vit_b16_backbone(
    model,
    encoder_feature_dims,
    encoder_feature_layer_ids,
    vit_features,
    start_index=1,
    use_grad_checkpointing=False,
) -> torch.nn.Module:
    """Make a ViTb16 backbone for the DPT model."""
    if use_grad_checkpointing:
        model.set_grad_checkpointing()

    vit_model = torch.nn.Module()
    vit_model.hooks = encoder_feature_layer_ids
    vit_model.model = model
    vit_model.features = encoder_feature_dims
    vit_model.vit_features = vit_features
    vit_model.model.start_index = start_index
    vit_model.model.patch_size = vit_model.model.patch_embed.patch_size
    vit_model.model.is_vit = True
    vit_model.model.forward = vit_model.model.forward_features

    return vit_model


def forward_features_eva_fixed(self, x):
    """Encode features."""
    x = self.patch_embed(x)
    x, rot_pos_embed = self._pos_embed(x)
    for blk in self.blocks:
        if self.grad_checkpointing:
            x = checkpoint(blk, x, rot_pos_embed)
        else:
            x = blk(x, rot_pos_embed)
    x = self.norm(x)
    return x


def resize_vit(
    model: torch.nn.Module, 
    img_size
) -> torch.nn.Module:
    """Resample the ViT module to the given size."""
    patch_size = model.patch_embed.patch_size
    model.patch_embed.img_size = img_size
    grid_size = tuple([s // p for s, p in zip(img_size, patch_size)])
    model.patch_embed.grid_size = grid_size

    pos_embed = resample_abs_pos_embed(
        model.pos_embed,
        grid_size,  # img_size
        num_prefix_tokens=(
            0 if getattr(model, "no_embed_class", False) else model.num_prefix_tokens
        ),
    )
    model.pos_embed = torch.nn.Parameter(pos_embed)

    return model


def resize_patch_embed(
    model: torch.nn.Module, 
    new_patch_size=(16, 16)
) -> torch.nn.Module:
    """Resample the ViT patch size to the given one."""
    # interpolate patch embedding
    if hasattr(model, "patch_embed"):
        old_patch_size = model.patch_embed.patch_size

        if (
            new_patch_size[0] != old_patch_size[0]
            or new_patch_size[1] != old_patch_size[1]
        ):
            patch_embed_proj = model.patch_embed.proj.weight
            patch_embed_proj_bias = model.patch_embed.proj.bias
            use_bias = True if patch_embed_proj_bias is not None else False
            _, _, h, w = patch_embed_proj.shape

            new_patch_embed_proj = torch.nn.functional.interpolate(
                patch_embed_proj,
                size=[new_patch_size[0], new_patch_size[1]],
                mode="bicubic",
                align_corners=False,
            )
            new_patch_embed_proj = (
                new_patch_embed_proj * (h / new_patch_size[0]) * (w / new_patch_size[1])
            )

            model.patch_embed.proj = torch.nn.Conv2d(
                in_channels=model.patch_embed.proj.in_channels,
                out_channels=model.patch_embed.proj.out_channels,
                kernel_size=new_patch_size,
                stride=new_patch_size,
                bias=use_bias,
            )

            if use_bias:
                model.patch_embed.proj.bias = patch_embed_proj_bias

            model.patch_embed.proj.weight = torch.nn.Parameter(new_patch_embed_proj)

            model.patch_size = new_patch_size
            model.patch_embed.patch_size = new_patch_size
            model.patch_embed.img_size = (
                int(
                    model.patch_embed.img_size[0]
                    * new_patch_size[0]
                    / old_patch_size[0]
                ),
                int(
                    model.patch_embed.img_size[1]
                    * new_patch_size[1]
                    / old_patch_size[1]
                ),
            )

    return model


def create_vit(
    preset: ViTPreset,
    use_pretrained: bool = False,
    checkpoint_uri: str | None = None,
    use_grad_checkpointing: bool = False,
) -> torch.nn.Module:
    """Create and load a VIT backbone module.

    Args:
    ----
        preset: The VIT preset to load the pre-defined config.
        use_pretrained: Load pretrained weights if True, default is False.
        checkpoint_uri: Checkpoint to load the wights from.
        use_grad_checkpointing: Use grandient checkpointing.

    Returns:
    -------
        A Torch ViT backbone module.

    """
    config = VIT_CONFIG_DICT[preset]

    img_size = (config.img_size, config.img_size)
    patch_size = (config.patch_size, config.patch_size)

    if "eva02" in preset:
        model = timm.create_model(config.timm_preset, pretrained=use_pretrained)
        model.forward_features = types.MethodType(forward_features_eva_fixed, model)
    else:
        model = timm.create_model(
            config.timm_preset, pretrained=use_pretrained, dynamic_img_size=True
        )
    model = make_vit_b16_backbone(
        model,
        encoder_feature_dims=config.encoder_feature_dims,
        encoder_feature_layer_ids=config.encoder_feature_layer_ids,
        vit_features=config.embed_dim,
        use_grad_checkpointing=use_grad_checkpointing,
    )
    if config.patch_size != config.timm_patch_size:
        model.model = resize_patch_embed(model.model, new_patch_size=patch_size)
    if config.img_size != config.timm_img_size:
        model.model = resize_vit(model.model, img_size=img_size)

    if checkpoint_uri is not None:
        state_dict = torch.load(checkpoint_uri, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict=state_dict, strict=False
        )

        if len(unexpected_keys) != 0:
            raise KeyError(f"Found unexpected keys when loading vit: {unexpected_keys}")
        if len(missing_keys) != 0:
            raise KeyError(f"Keys are missing when loading vit: {missing_keys}")

    LOGGER.info(model)
    return model.model


def create_backbone_model(
    preset: ViTPreset
) -> Tuple[torch.nn.Module, ViTPreset]:
    """Create and load a backbone model given a config.

    Args:
    ----
        preset: A backbone preset to load pre-defind configs.

    Returns:
    -------
        A Torch module and the associated config.

    """
    if preset in VIT_CONFIG_DICT:
        config = VIT_CONFIG_DICT[preset]
        model = create_vit(preset=preset, use_pretrained=False)
    else:
        raise KeyError(f"Preset {preset} not found.")

    return model, config


def load_config(config_fname):
    with open(config_fname, "r") as f:
        config = box.Box(yaml.safe_load(f))

    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        config.accelerator = "gpu"
        config.n_devices = n_gpus
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config.accelerator = "cpu"
        config.n_devices = 1
        config.device = 'cpu'

    return config


@pl.utilities.rank_zero_only
def zip_code(save_dir):
    os.system(f"zip {save_dir}/code.zip *.py config.yml")


def log_transform(tsdf):
    result = torch.log(tsdf.abs() + 1)
    result *= torch.sign(tsdf)
    return result


def tsdf2mesh(tsdf, voxel_size, origin, level=0):
    verts, faces, _, _ = skimage.measure.marching_cubes(tsdf, level=level)
    faces = faces[~np.any(np.isnan(verts[faces]), axis=(1, 2))]
    verts = verts * voxel_size + origin
    return trimesh.Trimesh(verts, faces)


def project(xyz, poses, K, imsize):
    """
    xyz: b x (*spatial_dims) x 3
    poses: b x nviews x 4 x 4
    K: (b x nviews x 3 x 3)
    imsize: (imheight, imwidth)
    """

    device = xyz.device
    batch_size = xyz.shape[0]
    spatial_dims = xyz.shape[1:-1]
    n_views = poses.shape[1]

    xyz = xyz.view(batch_size, 1, -1, 3).transpose(3, 2)
    xyz = torch.cat((xyz, torch.ones_like(xyz[:, :, :1])), dim=2)

    with torch.autocast(enabled=False, device_type=device.type):
        xyz_cam = (torch.inverse(poses.float()) @ xyz)[:, :, :3]
        uv = K @ xyz_cam

    z = uv[:, :, 2]
    uv = uv[:, :, :2] / uv[:, :, 2:]
    imheight, imwidth = imsize
    """
    assuming that these uv coordinates have
        (0, 0) = center of top left pixel
        (w - 1, h - 1) = center of bottom right pixel
    then we allow values between (-.5, w-.5) because they are inside the border pixel
    """
    valid = (
        (uv[:, :, 0] >= -0.5)
        & (uv[:, :, 1] >= -0.5)
        & (uv[:, :, 0] <= imwidth - 0.5)
        & (uv[:, :, 1] <= imheight - 0.5)
        & (z > 0)
    )
    uv = uv.transpose(2, 3)

    uv = uv.view(batch_size, n_views, *spatial_dims, 2)
    z = z.view(batch_size, n_views, *spatial_dims)
    valid = valid.view(batch_size, n_views, *spatial_dims)
    return uv, z, valid


def sample_posed_images(
    imgs, 
    poses, 
    K, 
    xyz, 
    mode="bilinear", 
    padding_mode="zeros", 
    return_z=False
):
    """
    imgs: b x nviews x C x H x W
    poses: b x nviews x 4 x 4
    K: (b x nviews x 3 x 3)
    xyz: b x (*spatial_dims) x 3
    """

    device = imgs.device
    batch_size, n_views, _, imheight, imwidth = imgs.shape
    spatial_dims = xyz.shape[1:-1]

    """
    assuming that these uv coordinates have
        (0, 0) = center of top left pixel
        (w - 1, h - 1) = center of bottom right pixel

    adjust because grid_sample(align_corners=False) assumes
        (0, 0) = top left corner of top left pixel
        (w, h) = bottom right corner of bottom right pixel
    """
    uv, z, valid = project(xyz, poses, K, (imheight, imwidth))
    imsize = torch.tensor([imwidth, imheight], device=device)
    # grid = (uv + 0.5) / imsize * 2 - 1
    grid = uv / (0.5 * imsize) + (1 / imsize - 1)
    vals = torch.nn.functional.grid_sample(
        imgs.view(batch_size * n_views, *imgs.shape[2:]),
        grid.view(batch_size * n_views, 1, -1, 2),
        align_corners=False,
        mode=mode,
        padding_mode=padding_mode,
    )
    vals = vals.view(batch_size, n_views, -1, *spatial_dims)
    if return_z:
        return vals, valid, z
    else:
        return vals, valid

def sample_voxel_feats(
    img_feats, 
    poses, 
    K, 
    xyz, 
    imsize, 
    invalid_fill_value=0
):
    base_imheight, base_imwidth = imsize
    featheight = img_feats.shape[3]
    featwidth = img_feats.shape[4]
    _K = K.clone()
    _K[:, :, 0] *= featwidth / base_imwidth
    _K[:, :, 1] *= featheight / base_imheight

    voxel_feats, valid = sample_posed_images(
        img_feats,
        poses,
        _K,
        xyz,
        mode="bilinear",
        padding_mode="border",
    )
    voxel_feats.masked_fill_(~valid[:, :, None], invalid_fill_value)

    return voxel_feats, valid


def density_fusion(
    voxel_size, 
    pred_depth_imgs, 
    poses, 
    K_pred_depth, 
    input_coords
):
    depth, valid, z = sample_posed_images(
        pred_depth_imgs[:, :, None],
        poses,
        K_pred_depth,
        input_coords,
        mode="nearest",
        return_z=True,
    )
    depth = depth.squeeze(2)
    valid.masked_fill_(depth == 0, False)

    dist = (z - depth).abs()
    in_voxel = valid & (dist < np.sqrt(3) * voxel_size / 2)

    weight = valid.sum(dim=1)
    density = in_voxel.sum(dim=1) / (weight + (weight == 0).to(weight.dtype))

    return density, weight


def tsdf_fusion(
    voxel_size, 
    pred_depth_imgs, 
    poses, 
    K_pred_depth, 
    input_coords
):
    depth, valid, z = sample_posed_images(
        pred_depth_imgs[:, :, None],
        poses,
        K_pred_depth,
        input_coords,
        mode="nearest",
        return_z=True,
    )
    depth = depth.squeeze(2)
    valid.masked_fill_(depth == 0, False)
    margin = 3 * voxel_size
    tsdf = torch.clamp(z - depth, -margin, margin) / margin
    valid &= tsdf < 0.999
    tsdf.masked_fill_(~valid, 0)
    tsdf = torch.sum(tsdf, dim=1)
    weight = torch.sum(valid, dim=1)
    tsdf /= weight
    return tsdf, weight


def sample_point_features_by_linear_interp(
    device, 
    voxel_size, 
    coords, 
    voxel_feats, 
    voxel_valid, 
    grid_origin
):
    """
    coords: BN3
    voxel_feats: BFXYZ
    voxel_valid: BXYZ
    grid_origin: B3
    """
    crop_size_m = (
        torch.tensor(voxel_feats.shape[2:], device=device)
        * voxel_size
    )
    grid = (
        coords - grid_origin[:, None] + voxel_size / 2
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


def augment_depth_inplace(
    device, 
    batch
):
    n_views = batch["depth"].shape[1]
    n_augment = n_views // 2

    for i in range(len(batch["depth"])):
        j = np.random.choice(
            batch["depth"].shape[1], size=n_augment, replace=False
        )
        scale = torch.rand(len(j), device=device) * 0.2 + 0.9
        batch["depth"][i, j] *= scale[:, None, None]


def compute_recon_loss(
    tsdf_logits,
    occ_logits,
    gt_tsdf,
    gt_occ,
    coarse_point_valid,
    fine_point_valid,
):
    occ_loss_mask = (~gt_occ.isnan()) & coarse_point_valid
    tsdf_loss_mask = (gt_occ > 0.5) & (~gt_tsdf.isnan()) & fine_point_valid

    occ_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        occ_logits[occ_loss_mask], gt_occ[occ_loss_mask]
    )

    loss = occ_loss
    if tsdf_loss_mask.sum() > 0:
        tsdf_loss = torch.nn.functional.l1_loss(
            log_transform(torch.tanh(tsdf_logits[tsdf_loss_mask])),
            log_transform(gt_tsdf[tsdf_loss_mask]),
        )
        loss += tsdf_loss
    else:
        tsdf_loss = torch.tensor(torch.nan)

    return loss, tsdf_loss, occ_loss


def recon_step(model, batch):
    device = model.device
    voxel_size = model.config.voxel_size
    if model.depth_guidance.enabled:
        if model.training and model.depth_guidance.depth_scale_augmentation:
            model.augment_depth_inplace(batch)

        voxel_feats, voxel_valid = model.get_img_voxel_feats_by_depth_guided_bp(
            batch["rgb_imgs"],
            batch["pred_depth_imgs"],
            batch["poses"],
            batch["K_color"][:, None],
            batch["K_pred_depth"][:, None],
            batch["input_coords"],
        )
        voxel_feats = model.fusion(voxel_feats, voxel_valid)
        voxel_valid = voxel_valid.sum(dim=1) > 1
        if model.config.no_image_features:
            voxel_feats = voxel_feats * 0

        if model.depth_guidance.density_fusion_channel:
            density, weight = density_fusion(
                voxel_size,
                batch["pred_depth_imgs"],
                batch["poses"],
                batch["K_pred_depth"][:, None],
                batch["input_coords"],
            )
            voxel_feats = torch.cat((voxel_feats, density[:, None]), dim=1)
        elif model.depth_guidance.tsdf_fusion_channel:
            tsdf, weight = tsdf_fusion(
                voxel_size,
                batch["pred_depth_imgs"],
                batch["poses"],
                batch["K_pred_depth"][:, None],
                batch["input_coords"],
            )
            tsdf.masked_fill_(weight == 0, 1)
            voxel_feats = torch.cat((voxel_feats, tsdf[:, None]), dim=1)
    else:
        voxel_feats, voxel_valid = model.get_img_voxel_feats_by_img_bp(
            batch["rgb_imgs"],
            batch["poses"],
            batch["K_color"][:, None],
            batch["input_coords"],
        )
        voxel_feats = model.fusion(voxel_feats, voxel_valid)
        voxel_valid = voxel_valid.sum(dim=1) > 1

    voxel_feats = model.cnn3d(voxel_feats, voxel_valid)

    if model.config.improved_tsdf_sampling:
        """
        interpolate the features to the points where we have GT tsdf
        """

        t = batch["crop_center"]
        R = batch["crop_rotation"]
        coords = batch["output_coords"]

        with torch.autocast(enabled=False, device_type=model.device.type):
            coords_local = (coords - t[:, None]) @ R
        coords_local += batch["crop_size_m"][:, None] / 2
        origin = torch.zeros_like(batch["gt_origin"])

        (
            coarse_point_feats,
            coarse_point_valid,
        ) = sample_point_features_by_linear_interp(
            device, voxel_size, coords_local, voxel_feats, voxel_valid, origin
        )
    else:
        """
        keep the voxel-center features that we have:
        GT tsdf has already been interpolated to these points
        """

        coarse_point_feats = voxel_feats.view(*voxel_feats.shape[:2], -1)
        coarse_point_valid = voxel_valid.view(voxel_valid.shape[0], -1)

    if model.config.point_backprojection:
        coords = batch["output_coords"]

        if model.depth_guidance.enabled:
            (
                fine_point_feats,
                fine_point_valid,
            ) = model.get_img_voxel_feats_by_depth_guided_bp(
                batch["rgb_imgs"],
                batch["pred_depth_imgs"],
                batch["poses"],
                batch["K_color"][:, None],
                batch["K_pred_depth"][:, None],
                coords,
                use_highres_cnn=True,
            )
        else:
            (
                fine_point_feats,
                fine_point_valid,
            ) = model.get_img_voxel_feats_by_img_bp(
                batch["rgb_imgs"],
                batch["poses"],
                batch["K_color"][:, None],
                coords,
                use_highres_cnn=True,
            )

        fine_point_feats = model.point_fusion(
            fine_point_feats[..., None, None], fine_point_valid[..., None, None]
        )[..., 0, 0]
        fine_point_valid = coarse_point_valid & (fine_point_valid.any(dim=1))
        fine_point_feats = model.point_feat_mlp(fine_point_feats)

        if model.config.no_image_features:
            fine_point_feats = fine_point_feats * 0

        if model.depth_guidance.enabled:
            if model.depth_guidance.density_fusion_channel:
                density, weight = model.density_fusion(
                    batch["pred_depth_imgs"],
                    batch["poses"],
                    batch["K_pred_depth"][:, None],
                    coords,
                )
                fine_point_feats = torch.cat(
                    (fine_point_feats, coarse_point_feats, density[:, None]), dim=1
                )
            elif model.depth_guidance.tsdf_fusion_channel:
                tsdf, weight = tsdf_fusion(
                    voxel_size,
                    batch["pred_depth_imgs"],
                    batch["poses"],
                    batch["K_pred_depth"][:, None],
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

    tsdf_logits = model.surface_predictor(fine_point_feats).squeeze(1)
    occ_logits = model.occ_predictor(coarse_point_feats).squeeze(1)

    loss, tsdf_loss, occ_loss = compute_recon_loss(
        tsdf_logits,
        occ_logits,
        batch["gt_tsdf"],
        batch["gt_occ"],
        coarse_point_valid,
        fine_point_valid,
    )

    outputs = {
        "loss": loss,
        "tsdf_loss": tsdf_loss,
        "occ_loss": occ_loss,
        "tsdf_logits": tsdf_logits,
        "occ_logits": occ_logits,
    }
    return outputs


def get_scans(config):
    train_scans, val_scans, test_scans = loaders.get_scans(
        config.dataset_dir,
        config.tsdf_dir,
        config.depth_guidance.pred_depth_dir,
    )
    return train_scans, val_scans, test_scans


def load_transform(config):
    transform = Compose(
        [
            ToTensor(),
            Lambda(lambda x: x.to(config.device)),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ConvertImageDtype(torch.float32),
        ]
    )
    return transform

def convert_to_gray_depth(
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
    
def generate_tsdf(
    voxel_size: np.float32,
    depths,
    poses, 
    intrinsic_depth
) -> dict[str, Any]:
    depth_height, depth_width = depths[0].shape
    u = np.arange(0, depth_width, 10)
    v = np.arange(0, depth_height, 10)
    uu, vv = np.meshgrid(u, v)
    uv = np.c_[uu.flatten(), vv.flatten()]
    pix_vecs = (np.linalg.inv(intrinsic_depth) @ np.c_[uv, np.ones((len(uv), 1))].T).T

    pts = []

    max_depth = None
    voxel_size = 0.02
    margin = int(np.round(0.04 / voxel_size))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for idx, local_depth in enumerate(depths):
        conv_depth = convert_to_gray_depth(local_depth, 0.99, None, np.float32)
        local_max_depth = conv_depth.max()
        if local_max_depth > max_depth:
            max_depth = local_max_depth

    for idx in range(len(poses)):
        pose = poses[idx]
        if np.any(np.isinf(pose)):
            continue

        depth = convert_to_gray_depth(depths[idx] / 1000, 0.99, None, depths[idx].dtype)
        depth = depth[uv[:, 1], uv[:, 0]]
        valid = depth > 0
        xyz_cam = pix_vecs[valid] * depth[valid, None]
        xyz = (pose @ np.c_[xyz_cam, np.ones((len(xyz_cam), 1))].T).T[:, :3]
        pts.append(xyz)

    pts = np.concatenate(pts, axis=0)

    minbound = np.min(pts, axis=0) - 3 * margin * voxel_size
    maxbound = np.max(pts, axis=0) + 3 * margin * voxel_size

    voxel_dim = torch.from_numpy(np.ceil((maxbound - minbound) / voxel_size)).int()
    origin = torch.from_numpy(minbound).float()

    torch.cuda.empty_cache()
    try:
        tsdf_vol = TSDFVolumeTorch(
            voxel_dim.to(device),
            origin.to(device),
            voxel_size,
            margin=margin,
            device=device,
        )
    except Exception as e:
        print(e)
        return
    
    for idx in len(poses):
        pose = poses[i]
        if np.any(np.isinf(pose)):
            continue
        depth = conv_depth = convert_to_gray_depth(depths[idx], 0.99, None, np.float32)
        depth[depth > max_depth] = 0
        tsdf_vol.integrate(
            torch.from_numpy(depth),
            torch.from_numpy(intrinsic_depth).float(),
            torch.from_numpy(pose).float(),
            1,
        )

    tsdf, weight = tsdf_vol.get_volume()

    tsdf[weight == 0] = torch.nan

    unobserved_col_mask = (
        (weight == 0).all(dim=-1, keepdim=True).repeat(1, 1, tsdf.shape[-1])
    )
    tsdf[unobserved_col_mask] = -1

    maxbound = origin + voxel_size * torch.tensor(tsdf.shape)

    return {
        'tsdf': tsdf.cpu().numpy(),
        'origin': origin.numpy(),   
        'voxel_size': voxel_size,
        'maxbound': maxbound.numpy(),
    }


lut = torch.tensor(plt.get_cmap("turbo").colors, dtype=torch.float32)  # (256,4)
lut = lut[:, :3]                                                       # (256,3)

@torch.compile
def convert_to_color_depth_lut(depth: torch.Tensor, dtype: torch.dtype=torch.torch.uint8) -> torch.Tensor:
    inverse_depth = 1 / depth
    
    max_invdepth_vizu = torch.min(
        inverse_depth.max(), 
        torch.tensor(1 / 0.1, device=depth.device)
    )
    min_invdepth_vizu = torch.max(
        torch.tensor(1 / 250, device=depth.device), 
        inverse_depth.min()
    )

    inverse_depth_normalized: torch.Tensor = (inverse_depth - min_invdepth_vizu) / (
        max_invdepth_vizu - min_invdepth_vizu
    )
    inverse_depth_normalized = inverse_depth_normalized.clamp(0.0, 1.0)
    
    idx = (inverse_depth_normalized * 255).round().long()
    color = lut.to(depth.device)[idx]

    return (color * 255).to(dtype=dtype)