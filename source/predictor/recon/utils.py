import os
import tqdm
import torch
import trimesh
import numpy as np
import skimage.measure
import pytorch_lightning as pl

from typing import Tuple, Optional
from numpy.typing import NDArray
from source.predictor.recon.tsdf_fusion import TSDFVolumeTorch

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
    imgs, poses, K, xyz, mode="bilinear", padding_mode="zeros", return_z=False
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


def sample_voxel_feats(img_feats, poses, K, xyz, imsize, invalid_fill_value=0):
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


def density_fusion(self, pred_depth_imgs, poses, K_pred_depth, input_coords, voxel_size):
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


def tsdf_fusion(pred_depth_imgs, poses, K_pred_depth, input_coords, voxel_size):
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


def estimate_volume_bounds_from_recon_datas(
    depths: NDArray,
    poses: NDArray,
    k_images: NDArray,
    max_depth = 3.5,
    voxel_size = 0.02,
    margin = None,
    device = None,
    ret_tsdf = False,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    margin = int(np.round(0.04 / voxel_size))
    _, _, imheight, imwidth = depths.shape

    K = k_images[:3, :3]
    
    u = np.arange(0, imwidth, 10)
    v = np.arange(0, imheight, 10)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    uv = np.c_[uu.flatten(), vv.flatten()]
    pix_vecs = (np.linalg.inv(K) @ np.c_[uv, np.ones((len(uv), 1))].T).T

    pts = []
    for i in tqdm.trange(0, len(poses), 10, leave=False, desc='computing scene bounds'):
        pose = poses[i]
        if np.any(np.isinf(pose)):
            continue
        depth = depths[i, 0, ...]
        depth[depth > max_depth] = 0
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
        ...

    for i in tqdm.trange(len(poses), leave=False, desc='TSDF fusion'):
        pose = poses[i]
        if np.any(np.isinf(pose)):
            continue
        depth = depths[i, 0, ...]
        depth[depth > max_depth] = 0
        tsdf_vol.integrate(
            torch.from_numpy(depth),
            torch.from_numpy(K).float(),
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

    return (
        tsdf if ret_tsdf else None,
        origin,
        maxbound,
    )