import os
import cv2
import sys
import box
import torch
import base64
import logging
import numpy as np
import open3d as o3d

from PIL import Image
from tqdm import tqdm
from numpy.typing import NDArray
from typing import cast, List, Dict, Tuple, Optional, Annotated
from open3d.geometry import TriangleMesh
from fastapi import status, Query, Body
from fastapi import APIRouter, Request, Depends
from fastapi.responses import Response, JSONResponse
from fastapi.exceptions import HTTPException

from source.utils import load_config, np_to_torch_dtype
from source.runtime.infer.defines import StoreStatus
from source.analyzer.predictor_analyzer import DelegateInstExecuter
from source.predictor.recon.predictor import ReconPredictor, ReconPro
from source.predictor.recon.data import ReconIterator
from source.predictor.ptv3.predictor import ExtractPredictor
from source.engine.arvane_engine import ArvaneEngine
from source.predictor.recon.data import (
    get_scans, 
    InferenceDataset, 
    transfer_batch_to_device
)

from source.predictor.utils import (
    split_mesh_by_vertex_color,
    points_and_colors_to_mesh,
    glb_bytes_to_o3d_mesh,
    trimesh_dict_to_o3d
)

from source.router.debug.debug_decl import DepthUpdatePayload
from source.router.debug.debug_impl import debug_start_reconstruction_impl
from source.engine.arvane_engine import ArvaneEngine
from source.runtime.device.device_manager import DeviceDescriptor
from source.runtime.infer.store import (
    InferenceChunkStoreConcurrent,
    EStoreObjectType,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_arvane_engine(request: Request) -> ArvaneEngine:
    return cast(ArvaneEngine, request.app.state.engine)

def get_arvane_container(request: Request) -> InferenceChunkStoreConcurrent:
    return cast(InferenceChunkStoreConcurrent, request.app.state.engine.container)

debug_router = APIRouter(prefix="/api/debug")

@debug_router.options("")
async def preflight() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@debug_router.post("/depth")
async def depth(
    payload: DepthUpdatePayload,
    arvane_engine: ArvaneEngine = Depends(get_arvane_engine),
) -> Response:
    task_id = payload.task_id
    buffer_b64 = payload.depth.buffer_b64
    depth = base64.b64decode(buffer_b64)
    depth = np.frombuffer(depth, dtype=np.uint8)
    depth = cv2.imdecode(depth, cv2.IMREAD_ANYDEPTH)
    depth = depth.astype(dtype=np.float32) / 1000
    
    arvane_engine.container.update_object_by_task_id(
        task_id,
        objects={
            'depth': (depth, )
        },
        tsk_type=EStoreObjectType.DEPTH,
        timestamp=payload.timestamp
    )

    return JSONResponse({ "result": "ok" }, status_code=status.HTTP_200_OK)

@debug_router.get("/detail")
async def debug_detail(
    arvane_engine: ArvaneEngine = Depends(get_arvane_engine)
) -> Response:
    ret_json: Dict = dict()
    num_gpus: int = arvane_engine.device_manager.gpu_num()
    for idx in range(num_gpus):
        descriptor: Optional[DeviceDescriptor] = arvane_engine.device_manager.gpu_desc(idx)
        if (descriptor is None):
            continue

        total_video_memory_per_mib: int = int(descriptor.memory.total_memory / (1024 ** 2))
        free_video_memory_per_mib: int = int(descriptor.memory.free_memory / (1024 ** 2))
        used_video_memory_per_mib: int = int(descriptor.memory.used_memory / (1024 ** 2))
        ret_json[str(idx)] = {
            "description": descriptor.description,
            "video_memory": {
                "total": f"{total_video_memory_per_mib} Mib",
                "free": f"{free_video_memory_per_mib} Mib",
                "used": f"{used_video_memory_per_mib} Mib"
            }
        }

    return JSONResponse(ret_json, status_code=status.HTTP_200_OK)

@debug_router.post("/predict/create")
async def debug_create_predictor(
    arvane_engine: ArvaneEngine = Depends(get_arvane_engine)
) -> Response:
    resp_json = {}
    user_id: str = "internal_debug_user"

    try:
        task_id = arvane_engine.container.add_task(user_id)
        resp_json['task_id'] = task_id

        logging.info("Debug: Created predictor task_id: %s", task_id)
        logging.info(
            f"Created world for user_id: {user_id}, task_id: {task_id}"
        )

        return JSONResponse(content=resp_json, status_code=status.HTTP_200_OK)
    
    except HTTPException as e:
        raise e

    except Exception as e:
        logging.error(f"Error creating world: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) from e

@debug_router.post("/predict/delete")
async def debug_delete_predictor(
    task_id: Annotated[str, Query(..., alias="task_id")],
    arvane_engine: ArvaneEngine = Depends(get_arvane_engine)
) -> Response:
    try:
        popped_store = arvane_engine.container.pop_task(task_id)

        image_len = len(popped_store.image_container)
        depth_len = len(popped_store.depth_container)
        pose_len  = len(popped_store.pose_container)
        k_image_len = len(popped_store.k_image_container)
        k_depth_len = len(popped_store.k_depth_container)

        logging.info(f"Deleted world for task_id: {task_id}, user_id: {popped_store.user_id}.")
        logging.info(f"image {image_len}, depth {depth_len}, pose {pose_len}, k_image {k_image_len}, k_depth {k_depth_len}.")
        return JSONResponse(content="ok", status_code=status.HTTP_200_OK)
    
    except HTTPException as e:
        raise e

    except Exception as e:
        logging.error(f"Error deleting predictor: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) from e

@debug_router.post("/predict/start/depth")
async def debug_start_predict_depths(
    task_id: Annotated[str, Body(..., alias="task_id", embed=True)],
    arvane_engine: ArvaneEngine = Depends(get_arvane_engine)
):
        if (not arvane_engine.container.exists(task_id)):
            logging.warning(f"Cannot find task_id: {task_id}, create new task_id before get world status.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Cannot find task_id, create new task_id before get world status."
            )

        logging.info(f"Initializing predictor config for task_id: {task_id}")
        recon_config: box.Box = arvane_engine.recon_predictor.config
        pred_depth_dir = "dataset/scannet_v2/dataset"
        dataset_dir = "dataset/scannet_v2/dataset"
        tsdf_dir = "dataset/scannet_v2/gt_tsdf"

        logging.info("Initializing predictor data scans.")
        test_scans: Tuple[List, List, List] = get_scans(
            dataset_dir=dataset_dir,
            tsdf_dir=tsdf_dir,
            pred_depth_dir=pred_depth_dir
        )[-1]

        logging.info("Preparing predictor dataset and dataloader.")
        dataset = InferenceDataset(scans=test_scans, load_depth=True, is_pred_depth=False)
        dataset_length = len(dataset)

        worker_predict = 1
        dataset_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=dataset_length,
            num_workers=worker_predict,
            persistent_workers=True,
            pin_memory=False,
            shuffle=False
        )
        
        # import pdb; pdb.set_trace()
        logging.info("Preparing batch for predictor inference.")
        batch = next(iter(dataset_loader))
        batch = transfer_batch_to_device(batch, recon_config.device)
        batch_iterator = ReconIterator(batch, enable_padding=True)

        TARGET_WIDTH, TARGET_HEIGHT = (640, 480)
        _, _, _, imheight, imwidth = batch["images"].shape
        gt_fx: float = float(batch["k_image"][0, 0, 0].detach().clone()) * (TARGET_WIDTH  / imwidth)
        gt_fy: float = float(batch["k_image"][1, 1, 0].detach().clone()) * (TARGET_HEIGHT / imheight)
        gt_f_px: float = float((gt_fx + gt_fy) / 2.0)

        for _, batch in tqdm(
            enumerate(batch_iterator), 
            desc="Predict depth from images"
        ):
            import pdb; pdb.set_trace()
            with torch.no_grad():
                depth, f_px = arvane_engine.depth_predictor.infer(
                    np.array(batch['images'][0, 0].permute(1, 2, 0)),
                    gt_f_px
                )

            depth: NDArray = cv2.resize(
                np.array(depth, dtype=np.float32), 
                (640, 480), 
                interpolation=cv2.INTER_AREA
            )

            arvane_engine.container[task_id].depth_container.add_object(depth)
            arvane_engine.container[task_id].focal_length_container.add_object(f_px)
        
@debug_router.post("/predict/start/recon")
async def debug_start_reconstruction(
    task_id: Annotated[str, Body(..., alias="task_id", embed=True)],
    arvane_engine: ArvaneEngine = Depends(get_arvane_engine)
) -> Response:
    if not arvane_engine.container.exists(task_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cannot find task_id.",
        )

    if arvane_engine.proxy_executor.is_running(task_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Reconstruction is already running.",
        )

    try:
        arvane_engine.proxy_executor.submit(
            task_id,
            debug_start_reconstruction_impl,
            task_id,
            arvane_engine,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "task_id": task_id,
            "status": "queued",
            "pending_tasks": arvane_engine.proxy_executor.num_pending_tasks,
            "max_workers": arvane_engine.proxy_executor.max_workers,
        },
    )

@debug_router.post("/predict/start/extraction")
async def debug_start_extraction(
    task_id: Annotated[str, Body(..., alias="task_id", embed=True)],
    arvane_engine: ArvaneEngine = Depends(get_arvane_engine)
):
    try:
        dataset_dir = "dataset/labels/scene0708_00/scene0708_00_vh_clean.ply"
        
        logging.info(f"Initializing predictor config for task_id: {task_id}")
        with torch.no_grad():
            points, color = arvane_engine.extract_predictor.infer(
                glb_bytes_or_file=dataset_dir, 
                return_color=True
            )

        logging.info(f"Prediction done, task_id: {task_id}")
        arvane_engine.container[task_id].store_status = StoreStatus.EXTRACT
        recon_object_considering_extraction: TriangleMesh = points_and_colors_to_mesh(points, color, 16)
        arvane_engine.container[task_id].recon_object = recon_object_considering_extraction
        
        # user_id = arvane_engine.container[task_id].user_id

        # target_path = f"logs/{user_id}/extraction/"
        # os.makedirs(target_path, exist_ok=True)
        # o3d.io.write_triangle_mesh(
        #     os.path.join(target_path, f"{task_id}.glb"), 
        #     recon_object_considering_extraction
        # )

        extraction_mesh: List[TriangleMesh] = split_mesh_by_vertex_color(
            recon_object_considering_extraction,
            color_eps=0.0001,
        )
        arvane_engine.container[task_id].extraction_object = extraction_mesh
        arvane_engine.container[task_id].store_status = StoreStatus.DONE

        return JSONResponse(
            content="ok", 
            status_code=status.HTTP_200_OK
        )
    
    except HTTPException as e:
        raise e

    except Exception as e:
        logging.error(f"Error starting predictor: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) from e