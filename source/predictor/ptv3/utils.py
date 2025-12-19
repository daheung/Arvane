"""
General utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import random
import numpy as np
import open3d as o3d
import torch
import torch.backends.cudnn as cudnn
import trimesh

from packaging import version
from typing import Optional, Dict, Tuple, Any
from open3d.geometry import TriangleMesh
from datetime import datetime
from numpy.typing import NDArray
from huggingface_hub import hf_hub_download

@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.no_grad()
def bincount2offset(bincount):
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.no_grad()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def get_random_seed():
    seed = (
        os.getpid()
        + int(datetime.now().strftime("%S%f"))
        + int.from_bytes(os.urandom(2), "big")
    )
    return seed


def set_seed(seed=None):
    if seed is None:
        seed = get_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

def extract_point_from_file(glb_path: str, fill_uv: Optional[Tuple[int, int, int]]) -> Dict[str, NDArray]:
    mesh: TriangleMesh = o3d.io.read_triangle_mesh(glb_path)
    if (mesh is None):
        raise ValueError(f"Mesh is empty. Check your glb path: {glb_path}")
    
    return extract_point_from_glb(mesh, fill_uv=fill_uv)

def extract_point_from_glb(mesh: TriangleMesh, fill_uv: Optional[Tuple[int, int, int]]) -> Dict[str, NDArray]:
    # glTF/GLB의 경우 좌표계/재질 후처리를 원하면 enable_post_processing 옵션을 시도해보세요:
    # mesh = o3d.io.read_triangle_mesh(glb_path, enable_post_processing=True)
    mesh.compute_vertex_normals()

    color = _extract_color(mesh, fill_uv)
    coord = _extract_coord(mesh)
    normal = _extract_normal(mesh)

    if (color is None):
        raise ValueError("Color must be not none.")

    return {
        "color": color,
        "coord": coord,
        "normal": normal
    }

def _extract_color(mesh, fill_uv: Optional[Tuple[int, int, int]]) -> Optional[NDArray]:
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors).astype(np.float32)
        return colors

    # 정점 색이 없으면 텍스처+UV로 베이크 시도
    if len(mesh.textures) > 0 and len(mesh.triangle_uvs) > 0:
        baked = _bake_vertex_colors_from_uv(mesh)
        return baked

    if (fill_uv is not None):
        return np.full((len(mesh.vertices), 3), fill_uv)

    return None

def _bake_vertex_colors_from_uv(mesh: TriangleMesh) -> NDArray:
    """
    triangle_uvs(각 삼각형 3개 코너당 1개씩)와 텍스처 1장을 이용해
    각 정점의 uv를 인접 코너 uv들의 평균으로 근사하여 정점 색을 베이크.
    반환: (N_vertices, 3) in [0,1]
    """
    if len(mesh.textures) == 0 or len(mesh.triangle_uvs) == 0:
        raise ValueError("Cannot bake due to texture or triangle_uvs is None.")

    # 단일 텍스처 가정
    tex_img_o3d = mesh.textures[0]
    tex_np = np.asarray(tex_img_o3d)  # (H, W, 3|4) uint8

    triangles = np.asarray(mesh.triangles)            # (T, 3)
    tri_uvs   = np.asarray(mesh.triangle_uvs)         # (T*3, 2), 삼각형 순서대로 코너 3개씩
    n_verts   = np.asarray(mesh.vertices).shape[0]

    # 정점별로 인접 코너 UV 수집 -> 평균
    uv_sum   = np.zeros((n_verts, 2), dtype=np.float64)
    uv_count = np.zeros((n_verts,), dtype=np.int64)

    # 각 삼각형 t의 코너 k(0,1,2)에 대응하는 tri_uvs 인덱스는 t*3 + k
    for t in range(triangles.shape[0]):
        vidx = triangles[t]             # (3,)
        uvs3 = tri_uvs[t*3:(t+1)*3]     # (3, 2)
        for k in range(3):
            v_id = vidx[k]
            uv_sum[v_id]   += uvs3[k]
            uv_count[v_id] += 1

    # 고립 정점 방지
    uv_count[uv_count == 0] = 1
    vert_uv = uv_sum / uv_count[:, None]   # (N, 2), 범위 보통 [0,1]

    # 텍스처에서 샘플
    vert_rgb = _sample_texture(tex_np, vert_uv)  # (N, 3) float
    return vert_rgb

def _sample_texture(img_np: np.ndarray, uv: np.ndarray) -> NDArray:
    """
    img_np: (H, W, 3|4) uint8
    uv: (..., 2) with range [0, 1], (u, v) where v-axis is top->bottom in glTF
    returns: (..., 3) float in [0,1]
    """
    H, W = img_np.shape[:2]
    # glTF의 v는 이미지 top->bottom 기준. Pillow/NumPy도 (0,0)이 top-left이므로
    # 보통 v 그대로 써도 되지만, 일부 파이프라인은 v를 뒤집어야 할 수 있습니다.
    # 필요시 v = 1 - v 로 바꿔보세요.
    u = np.clip(uv[..., 0], 0, 1) * (W - 1)
    v = np.clip(uv[..., 1], 0, 1) * (H - 1)

    # bilinear 보간 대신 최근접 샘플(간단)
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)

    rgb = img_np[vi, ui, :3].astype(np.float32) / 255.0
    return rgb

def _extract_coord(mesh: TriangleMesh) -> NDArray:
    if not mesh.has_vertices():
        raise ValueError("Cannot found vertices in mesh. Check your path or mesh data.")
    
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    return verts

def _extract_normal(mesh: TriangleMesh) -> NDArray:
    if not mesh.has_vertices():
        raise RuntimeError("Mesh is empty.")
    
    # 1) 정점 노멀 (있으면 그대로 사용, 없으면 계산)
    if not mesh.has_vertex_normals():
        # 면 노멀 먼저 계산 후, 인접 면 평균으로 정점 노멀을 만듭니다.
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
    
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    return vertex_normals

MODELS = [
    "sonata",
    "sonata_small",
    "sonata_linear_prob_head_sc",
]

def load_ckpt(
    name: str = "sonata",
    repo_id="facebook/sonata",
    download_root: str = None,
    custom_config: dict = None,
) -> Any:
    if name in MODELS:
        print(f"Loading checkpoint from HuggingFace: {name} ...")
        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{name}.pth",
            repo_type="model",
            revision="main",
            local_dir=download_root or os.path.expanduser("~/.cache/sonata/ckpt"),
        )
    elif os.path.isfile(name):
        print(f"Loading checkpoint in local path: {name} ...")
        ckpt_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {MODELS}")

    if version.parse(torch.__version__) >= version.parse("2.4"):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    if custom_config is not None:
        for key, value in custom_config.items():
            ckpt["config"][key] = value

    return ckpt

def load(
    name: str = "sonata",
    repo_id="facebook/sonata",
    download_root: str = None,
    custom_config: dict = None,
):
    ckpt = load_ckpt(
        name=name,
        repo_id=repo_id,
        download_root=download_root,
        custom_config=custom_config,
    )
    
    from source.predictor.ptv3.predictor import PointTransformerV3

    model = PointTransformerV3(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_parameters / 1e6:.2f}M")
    return model
