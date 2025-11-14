import cv2
import sys
import time
import torch
import logging
import asyncio
import numpy as np
import base64, io

from PIL import Image
from typing import Optional, Annotated, cast
from fastapi import status, Query
from fastapi import BackgroundTasks, APIRouter, Request, Depends
from fastapi.responses import Response, JSONResponse
from fastapi.exceptions import HTTPException
from numpy.typing import NDArray

from .world_decl import (
    WorldCreatePayload,
    WorldDeletePayload, 
    WorldUpdatePayload,
    WorldStartPayload,
    verify_world_update
)

from source.router.utils.utils import NDArrayB64
from source.runtime.infer.defines import ReconLog
from source.runtime.infer.store import (
    EStoreObjectType,
    EStoreOperatorType
)

from source.engine.arvane_engine import ArvaneEngine
from source.runtime.infer.store import InferenceChunkStoreConcurrent

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

world_router = APIRouter(prefix="/api/world")

@world_router.options("")
async def preflight() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@world_router.post("/create")
async def world_status(
    payload: WorldCreatePayload,
    arvane_engine: ArvaneEngine = Depends(get_arvane_engine),
):  
    resp_json = {}
    try:
        user_id: Optional[str] = payload.user_id
        if (user_id is None) or (user_id.strip() == ""):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required"
            )

        task_id = arvane_engine.container.add_task(user_id)
        resp_json['task_id'] = task_id

        logging.info(
            f"Created world for user_id: {user_id}, task_id: {task_id}, name: {payload.name}"
        )

        return JSONResponse(content=resp_json, status_code=status.HTTP_200_OK)
    
    except Exception as e:
        logging.error(f"Error creating world: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) from e

@world_router.post("/delete")
async def world_delete(
    payload: WorldDeletePayload,
    arvane_engine: ArvaneEngine = Depends(get_arvane_engine),
):
    try:
        popped_store = arvane_engine.container.pop_task(payload.task_id)

        image_len = len(popped_store.image_container)
        depth_len = len(popped_store.depth_container)
        pose_len  = len(popped_store.pose_container)
        k_image_len = len(popped_store.k_image_container)
        k_depth_len = len(popped_store.k_depth_container)

        logging.info(f"Deleted world for task_id: {payload.task_id}, user_id: {popped_store.user_id}, name: {payload.name}.")
        logging.info(f"image {image_len}, depth {depth_len}, pose {pose_len}, k_image {k_image_len}, k_depth {k_depth_len}.")

        return JSONResponse(content="ok", status_code=status.HTTP_200_OK)
    
    except KeyError as ke:
        logging.error(f"Error deleting world: {ke}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ke)
        ) from ke

    except Exception as e:
        logging.error(f"Error deleting world: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) from e

@world_router.post("/update")
async def world_update(
    payload: WorldUpdatePayload,
    arvane_engine: ArvaneEngine = Depends(get_arvane_engine),
):
    try:
        # check payload validity. if invalid, raise ValueError
        verify_world_update(payload)
        
        task_id = payload.task_id
        buffer_b64 = payload.color.buffer_b64
        if payload.color.data_url:
            _, buffer_b64 = buffer_b64.split(",", 1)

        color = base64.b64decode(buffer_b64)
        color = Image.open(io.BytesIO(color)).convert("RGB")
        color = np.array(color).transpose((2, 0, 1))

        k_color: NDArray = np.array(payload.k_color, dtype=np.float32).reshape((3, 3))
        pose   : NDArray = np.array(payload.pose   , dtype=np.float32).reshape((4, 4))

        arvane_engine.container.update_object_by_task_id(
            task_id,
            objects={
                'image': (color, ),
                'k_image': (k_color, ),
                'k_depth': (k_color, ),
                'pose': (pose, )
            },
            tsk_type=EStoreObjectType.RECON_INFER_NO_DEPTH,
            op_type=EStoreOperatorType.INSERT
        )

        if (payload.auto_update_depth):
            logging.info(f"Start auto-update depth for task_id: {task_id}")
            arvane_engine.update_depth_object_by_task_id(task_id)
            return JSONResponse(
                content= {
                    "result": "ok", 
                    "task_id": task_id
                }, 
                status_code=status.HTTP_202_ACCEPTED
            )

        logging.info(f"Updated world for task_id: {task_id}")
        return JSONResponse(content="ok", status_code=status.HTTP_200_OK)

    except ValueError as ve:
        logging.error(f"Invalid update payload: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        ) from ve
    
    except Exception as e:
        logging.error(f"Error updating world: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) from e



@world_router.post('/start')
async def world_start(
    payload: WorldStartPayload,
    arvane_engine: ArvaneEngine = Depends(get_arvane_engine),
):
    try:
        asyncio.create_task(
            arvane_engine.run_process(payload.task_id)
        )

        return JSONResponse(
            content={
                "result": "accepted",
                "task_id": payload.task_id,
                "message": "reconstruction started in background",
            },
            status_code=status.HTTP_202_ACCEPTED,
        )
    
    except HTTPException as he:
        raise he
    
    except Exception as e:
        logging.error(f"Error starting recon inference: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) from e
    
@world_router.get("/status")
async def world_status(
    task_id: Annotated[str, Query(..., alias="task_id")],
    arvane_engine: ArvaneEngine = Depends(get_arvane_engine),
):
    try:
        if (not arvane_engine.container.exists(task_id)):
            logging.warning(f"Cannot find task_id: {task_id}, create new task_id before get world status.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Cannot find task_id, create new task_id before get world status."
            )
        
        store = arvane_engine.container.load_object_by_task_id(
            task_id,
            tsk_type=EStoreObjectType.RECON_INFER_OBJECT
        )

        resp_json = {
            "num_image": len(store.get("image", [])),
            "num_depth": len(store.get("depth", [])),
            "num_pose" : len(store.get("pose" , [])),
            "num_k_image": len(store.get("k_image", [])),
            "num_k_depth": len(store.get("k_depth", [])),
        }

        logging.info(f"Status for task_id: {task_id} - {resp_json}")
        return JSONResponse(content=resp_json, status_code=status.HTTP_200_OK)
    
    except HTTPException as he:
        raise he
    
    except KeyError as ke:
        logging.error(f"Error retrieving world status: {ke}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ke)
        ) from ke

    except Exception as e:
        logging.error(f"Error retrieving world status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) from e
    

@world_router.get("/detail")
async def world_detail(
    task_id: Annotated[str, Query(..., alias="task_id")],
    arvane_engine: ArvaneEngine = Depends(get_arvane_engine)
):
    try:
        if (not arvane_engine.container.exists(task_id)):
            logging.warning(f"Cannot find task_id: {task_id}, create new task_id before get world status.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Cannot find task_id, create new task_id before get world status."
            )

        store = arvane_engine.container.load_object_by_task_id(
            task_id,
            tsk_type=EStoreObjectType.ALL_OBJECTS
        )

        recon_log: Optional[ReconLog] = store.get("recon_log", None)
        resp_json = {
            "num_image": len(store.get("image", [])),
            "num_depth": len(store.get("depth", [])),
            "num_pose" : len(store.get("pose" , [])),
            "num_k_image": len(store.get("k_image", [])),
            "num_k_depth": len(store.get("k_depth", [])),
            "recon": {
                "start_init_time": recon_log.init_time_0,
                "end_init_time": recon_log.init_time_1,
                "num_inits": recon_log.n_inits,
                "num_steps": recon_log.n_views,
                "start_final_time": recon_log.final_step_time_0,
                "start_final_time": recon_log.final_step_time_1,
                "per_view_time": recon_log.per_view_time,
            }
        }

        return JSONResponse(resp_json, status_code=status.HTTP_200_OK)

    except HTTPException as e:
        ...

    except Exception as e:
        logging.error(f"Error retrieving world status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) from e
    

@world_router.get("/result")
async def world_result(
    task_id: Annotated[str, Query(..., alias="task_id")],
    arvane_engine: ArvaneEngine = Depends(get_arvane_engine)
):
    try:
        if (not arvane_engine.container.exists(task_id)):
            logging.warning(f"Cannot find task_id: {task_id}, create new task_id before get world status.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Cannot find task_id, create new task_id before get world status."
            )

        result = arvane_engine.container.load_object_by_task_id(
            task_id=task_id,
            tsk_type=EStoreObjectType.RECON_RESULT | EStoreObjectType.RECON_STATUS
        )

        glb_bytes = result['recon_result']
        if (glb_bytes is None):
            return JSONResponse(
                content={ 
                    "status": f'inference {str(result['recon_status'])}.',
                }, 
                status_code=status.HTTP_202_ACCEPTED 
            )
        
        return Response(
            content=glb_bytes,
            media_type="model/gltf-binary",
            headers={
                "Content-Disposition": 'inline; filename="generated.glb"'
            }
        )

    except HTTPException as e:
        raise e
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )