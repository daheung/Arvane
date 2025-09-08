import cv2
import time
import anyio
import base64
import numpy as np

from typing import Tuple, Literal, Any, Callable
from fastapi import APIRouter, Request, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator


class NDArrayBuf(BaseModel):
    shape: Tuple[int, ...]
    dtype: str
    order: Literal["C", "F"] = "C"
    endian: Literal["Little", "Big"] = "Little"
    buffer: bytes = Field(..., description="raw bytes")

    @classmethod
    def from_ndarray(cls, arr: np.ndarray, order: Literal["C", "F"] = "C", endian: Literal["Little", "Big"] = "Little"):
        order_func = np.ascontiguousarray if order == "C" else np.asfortranarray
        arr = order_func(arr)
        
        desired = "<" if endian == "Little" else ">"
        arr = arr.astype(arr.dtype.newbyteorder(desired), copy=False)

        buffer = arr.tobytes(order=order)
        dtype_name = np.dtype(arr.dtype).name  # ex) 'float32'

        return cls(shape=arr.shape, dtype=dtype_name, order=order, endian=endian, buffer=buffer)
    
def bench_mark(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    start = time.time()
    ret = func(*args, **kwargs)
    end = time.time()

    return ret, (end - start)