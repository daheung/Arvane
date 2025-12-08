import os
import torch
import tempfile
import numpy as np
import open3d as o3d

from open3d.geometry import TriangleMesh
from numpy.typing import NDArray
from typing import Dict, Any, List, Union, Sequence
from collections import defaultdict

def pad_last(x: torch.Tensor, target_len: int, dim: int = 0) -> torch.Tensor:
    """
    x: tensor with shape (..., L, ...)
    dim: pad 대상 축
    """

    dim = dim % x.ndim
    L = x.size(dim)
    if L == 0:
        raise ValueError("해당 축 길이가 0이면 마지막 값을 복제할 수 없습니다.")
    if target_len <= L:
        # 슬라이스로 잘라서 반환
        sl = [slice(None)] * x.ndim
        sl[dim] = slice(0, target_len)
        return x[tuple(sl)]

    # 인덱스 만들기: [0,1,2,...,L-1,L-1,L-1,...] 형태
    idx = torch.arange(target_len, device=x.device).clamp_max(L - 1)

    # dim 축으로 선택
    return x.index_select(dim, idx)


def points_and_colors_to_mesh(
    points: torch.Tensor,   # (N, 3) float, world coords
    colors: torch.Tensor,   # (N, 3) uint8(0~255) 또는 float
    depth: int = 9,         # Poisson 재구성 깊이 (해상도 컨트롤)
) -> o3d.geometry.TriangleMesh:
    """
    torch.Tensor로 된 points, colors를 받아
    Open3D TriangleMesh를 생성해서 반환.
    """

    # 1) torch.Tensor -> numpy
    pts_np = points.detach().cpu().numpy().astype(np.float64)           # (N, 3)
    col_np = colors.detach().cpu().numpy().astype(np.float64) / 255.0   # (N, 3), [0,1]

    # 2) PointCloud 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)
    pcd.colors = o3d.utility.Vector3dVector(col_np)

    # 3) 노멀 추정 (Poisson 재구성에 필요)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05,  # 데이터 스케일에 맞게 조절
            max_nn=30,
        )
    )

    # 4) Poisson Surface Reconstruction으로 TriangleMesh 생성
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,    # 클수록 디테일 ↑, 메모리/시간 ↑
    )

    # 5) 너무 density 낮은 vertex 제거해서 노이즈 정리 (선택)
    densities = np.asarray(densities)
    density_thresh = np.quantile(densities, 0.01)  # 하위 1% 제거
    vertices_to_remove = densities < density_thresh
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # 6) 메쉬 vertex에 색깔 입히기
    #    → PointCloud에서 가장 가까운 포인트 색을 가져와서 전파
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    mesh_colors = []

    for v in mesh.vertices:
        # v: 3D vertex (Open3D Vector3d)
        _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)  # 가장 가까운 포인트 1개
        mesh_colors.append(col_np[idx[0]])

    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh_colors))

    # 7) 필요하면 노멀 재계산
    mesh.compute_vertex_normals()

    return mesh


def split_mesh_by_vertex_color(
    mesh: o3d.geometry.TriangleMesh,
    color_eps: float = 1e-5,
) -> List[o3d.geometry.TriangleMesh]:
    """
    하나의 TriangleMesh에서 vertex_colors를 기준으로
    (거의) 같은 색을 가진 삼각형들끼리 묶어서 여러 개의 sub-mesh로 분리한다.
    """

    if not mesh.has_vertex_colors():
        raise ValueError("mesh.vertex_colors 가 비어 있습니다. 색 정보가 필요합니다.")

    vertices = np.asarray(mesh.vertices)          # (V, 3)
    triangles = np.asarray(mesh.triangles)        # (F, 3) int
    colors = np.asarray(mesh.vertex_colors)       # (V, 3) float, 보통 [0,1]

    # 색을 key로 쓰기 위해 약간 quantize
    def color_key(c: np.ndarray):
        # eps 단위로 반올림해서 같은 색은 같은 tuple 이 되도록
        return tuple(np.round(c / color_eps).astype(int))

    # color_key -> triangle indices
    face_groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    for f_idx, (i0, i1, i2) in enumerate(triangles):
        c0, c1, c2 = colors[i0], colors[i1], colors[i2]
        # 세 정점 색이 거의 같다고 가정하고 평균을 대표 색으로 사용
        c_mean = (c0 + c1 + c2) / 3.0
        k = color_key(c_mean)
        face_groups[k].append(f_idx)

    sub_meshes: list[o3d.geometry.TriangleMesh] = []

    for k, face_indices in face_groups.items():
        face_indices = np.asarray(face_indices, dtype=np.int64)

        # 1) 이 color 그룹에 속한 삼각형들만 추출
        tris_sub = triangles[face_indices]        # (F_i, 3)

        # 2) 사용되는 vertex index만 unique 하게 뽑고, 새 index로 리매핑
        unique_vidx, inverse = np.unique(tris_sub.reshape(-1), return_inverse=True)
        new_vertices = vertices[unique_vidx]      # (V_i, 3)
        new_colors   = colors[unique_vidx]        # (V_i, 3)

        new_tris = inverse.reshape(-1, 3).astype(np.int32)  # (F_i, 3)

        # 3) 새로운 TriangleMesh 생성
        submesh = o3d.geometry.TriangleMesh()
        submesh.vertices      = o3d.utility.Vector3dVector(new_vertices)
        submesh.triangles     = o3d.utility.Vector3iVector(new_tris)
        submesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
        submesh.compute_vertex_normals()

        sub_meshes.append(submesh)

    return sub_meshes


def trimesh_dict_to_o3d(blob: Dict[str, Any]) -> o3d.geometry.TriangleMesh:
    vertices = np.asarray(blob["vertices"], dtype=np.float64)  # (N, 3)
    faces    = np.asarray(blob["faces"],    dtype=np.int64)    # (M, 3)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # 색 정보가 있으면
    if "vertex_colors" in blob:
        colors = np.asarray(blob["vertex_colors"])
        # 0~255 범위면 0~1로 normalize
        if colors.max() > 1.0:
            colors = colors / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3])

    # 노말 정보가 있으면
    if "vertex_normals" in blob:
        normals = np.asarray(blob["vertex_normals"], dtype=np.float64)
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    # 노말 없으면 새로 계산
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    return mesh

def glb_bytes_to_o3d_mesh(glb_bytes: bytes) -> o3d.geometry.TriangleMesh:
    with tempfile.NamedTemporaryFile(suffix=".glb") as f:
        f.write(glb_bytes)
        f.flush()
        mesh = o3d.io.read_triangle_mesh(f.name, enable_post_processing=True)
    return mesh


def o3d_mesh_to_glb_bytes(mesh: o3d.geometry.TriangleMesh) -> bytes:
    fd, tmp_path = tempfile.mkstemp(suffix=".glb")
    os.close(fd)

    try:
        success = o3d.io.write_triangle_mesh(
            tmp_path,
            mesh,
            write_vertex_normals=True,
            write_vertex_colors=True,
            write_triangle_uvs=True,
        )
        if not success:
            raise RuntimeError(f"Failed to write GLB mesh to {tmp_path}")

        with open(tmp_path, "rb") as f:
            data = f.read()
        return data
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def o3d_meshes_to_glb_bytes(
    mesh: Union[
        o3d.geometry.TriangleMesh,
        Sequence[o3d.geometry.TriangleMesh],
    ]
) -> bytes:
    """
    단일 TriangleMesh 또는 TriangleMesh 리스트를 받아 GLB 바이너리로 변환.
    """

    if isinstance(mesh, (list, tuple)):
        if len(mesh) == 0:
            raise ValueError("mesh list is empty")
        elif len(mesh) == 1:
            mesh_to_write = mesh[0]
        else:
            mesh_to_write = merge_triangle_meshes(mesh)
    else:
        mesh_to_write = mesh

    fd, tmp_path = tempfile.mkstemp(suffix=".glb")
    os.close(fd)

    try:
        success = o3d.io.write_triangle_mesh(
            tmp_path,
            mesh_to_write,
            write_vertex_normals=True,
            write_vertex_colors=True,
            write_triangle_uvs=True,
        )
        if not success:
            raise RuntimeError(f"Failed to write GLB mesh to {tmp_path}")

        with open(tmp_path, "rb") as f:
            data = f.read()

        return data

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def merge_triangle_meshes(
    meshes: Sequence[o3d.geometry.TriangleMesh],
) -> o3d.geometry.TriangleMesh:
    if not meshes:
        raise ValueError("mesh list is empty")

    merged = o3d.geometry.TriangleMesh()
    for m in meshes:
        merged += m

    merged.remove_duplicated_vertices()
    merged.remove_duplicated_triangles()
    merged.remove_unreferenced_vertices()

    return merged


def pixel_to_world(
    u: float,
    v: float,
    depth_map: np.ndarray,
    K: np.ndarray,
    T_cam2world: np.ndarray,
) -> np.ndarray:
    """
    (u, v) 픽셀과 depth, K, 카메라 pose를 이용해 world 좌표계의 3D 점을 구한다.

    Parameters
    ----------
    u, v : float
        이미지 좌표 (픽셀). u: x (width 방향), v: y (height 방향)
    depth_map : np.ndarray, shape (H, W)
        DepthPro가 예측한 metric depth (단위: meter)
    K : np.ndarray, shape (3, 3)
        카메라 내참수 행렬 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    T_cam2world : np.ndarray, shape (4, 4)
        카메라 좌표 -> 월드 좌표 변환 행렬 (camera-to-world pose)

    Returns
    -------
    np.ndarray, shape (3,)
        world 좌표계의 3D 점 (x, y, z)
    """
    H, W = depth_map.shape[:2]
    u_int = int(round(u))
    v_int = int(round(v))

    if not (0 <= u_int < W and 0 <= v_int < H):
        raise ValueError(f"pixel ({u}, {v}) is out of image bounds {W}x{H}")

    z = float(depth_map[v_int, u_int])  # depth_map[y, x]

    if z <= 0 or not np.isfinite(z):
        raise ValueError(f"Invalid depth value at ({u_int}, {v_int}): {z}")

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    pt_cam = np.array([x_cam, y_cam, z, 1.0], dtype=np.float32)

    pt_world_h = T_cam2world @ pt_cam
    pt_world = pt_world_h[:3] / pt_world_h[3]

    return pt_world


def distance_between_points_on_glb(
    uv1,
    uv2,
    depth_map: np.ndarray,
    K: np.ndarray,
    poses: np.ndarray,
    frame_idx: int = 0,
    pose_is_cam2world: bool = True,
) -> float:
    """
    pose.npy, DepthPro depth, intrinsic K를 이용해
    GLB 상 두 점(두 픽셀에 대응되는 3D 점) 사이의 거리를 구한다.

    Parameters
    ----------
    uv1, uv2 : tuple (u, v)
        첫 번째 / 두 번째 점의 이미지 좌표 (픽셀)
    depth_map : np.ndarray, shape (H, W)
        DepthPro의 metric depth (meters)
    K : np.ndarray, shape (3, 3)
        카메라 intrinsic matrix
    poses : np.ndarray, shape (N, 4, 4)
        pose.npy에서 로드한 카메라 pose 배열.
        frame_idx에 해당하는 pose를 사용.
    frame_idx : int, default 0
        사용할 프레임 index
    pose_is_cam2world : bool, default True
        True  : poses[frame_idx]가 camera-to-world T_c2w 라고 가정
        False : poses[frame_idx]가 world-to-camera T_w2c 라고 가정 (이 경우 inverse 해서 사용)

    Returns
    -------
    float
        두 3D 점 사이의 유클리드 거리 (meter)
    """
    T = poses[frame_idx]  # (4, 4)

    if not pose_is_cam2world:
        T = np.linalg.inv(T)

    u1, v1 = uv1
    u2, v2 = uv2

    p1_world = pixel_to_world(u1, v1, depth_map, K, T)
    p2_world = pixel_to_world(u2, v2, depth_map, K, T)

    dist = float(np.linalg.norm(p1_world - p2_world))
    return dist
