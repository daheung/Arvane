import cv2
import sys
import time
import logging
import numpy as np

from typing import cast
from fastapi import status
from fastapi import APIRouter, Request, Depends
from fastapi.responses import Response, JSONResponse
from fastapi.exceptions import HTTPException

from .utils import NDArrayBuf, bench_mark
from .verify import (
    check_dtype, 
    InvalidRequestQueryType, 
    InvalidRequestQueryException,
    InvalidRequestContentTypeException,
    InvalidRequestBodyException,
)

from ..predictor.depth.predictor import DepthPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

infer_router = APIRouter(prefix="/api/infer/depth")

def get_depth_inference(request: Request) -> DepthPredictor:
    return cast(DepthPredictor, request.app.state.depth_predictor)

@infer_router.options("")
async def preflight() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)
    
@infer_router.post("")
async def infer_depth(
    request: Request,
    depth_predictor: DepthPredictor = Depends(get_depth_inference)
):
    try:
        request_dtype = request.query_params.get("dtype") or "float32"
        if not (check_dtype(request_dtype)):
            raise InvalidRequestQueryException(InvalidRequestQueryType.IQT_Dtype, request_dtype)
        
        content_type = request.headers.get("content-type", "")
        if not (content_type.startswith("image/")):
            raise InvalidRequestContentTypeException(content_type)
        
        request_body = await request.body()
        raw_buf = np.frombuffer(request_body, dtype=np.uint8)
        rgb_img = cv2.imdecode(raw_buf, cv2.IMREAD_UNCHANGED)
        if (rgb_img is None):
            raise InvalidRequestBodyException('none')
        
        (depth, _), e_time = bench_mark(depth_predictor.infer, rgb_img)
        
        depth_arr = NDArrayBuf.from_ndarray(np.array(depth, dtype=request_dtype))
        depth_ser = depth_arr.buffer

        expose = ["Depth-Shape", "Depth-Dtype", "Depth-Endian", "Depth-Order", "Depth-Infer-Time"]
        headers = {
            "Depth-Shape": ",".join(map(str, depth_arr.shape)),
            "Depth-Dtype": depth_arr.dtype,
            "Depth-Endian": depth_arr.endian,
            "Depth-Order": depth_arr.order,
            "Depth-Infer-Time": str(e_time),
            "Access-Control-Expose-Headers": ", ".join(expose)
        }

        return Response(
            status_code=status.HTTP_200_OK,
            media_type="application/octet-stream",
            headers=headers, content=depth_ser
        )
    
    except InvalidRequestQueryException as e:
        logging.error(e)
        error_content = {
            "code": status.HTTP_400_BAD_REQUEST,
            "message": "Invalid dtype in query",
            "detail": str(e)
        }
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_content
        )
    
    except InvalidRequestBodyException as e:
        logging.error(e)
        error_content = {
            "code": status.HTTP_400_BAD_REQUEST,
            "message": str(e),
            "detail": "Cannot read image. check your body payload."
        }
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_content
        )

    except InvalidRequestContentTypeException as e:
        logging.error(e)
        error_content = {
            "code": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "message": "Unsupported Media Type",
            "detail": str(e)
        }
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content=error_content
        )

    except Exception as e:
        logging.error(e)
        error_content = {
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": "Unexpected Error",
            "detail": str(e)
        }
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_content
        )
    

update_router = APIRouter(prefix="/api/update/depth")

@update_router.post("{user_id}/create")
async def update_user_depth():
    pass