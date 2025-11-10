import cv2
import time
import anyio
import base64
import numpy as np

from typing import Tuple, Literal, Any, Callable, Union
from fastapi import APIRouter, Request, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

import numpy as np
from typing import Tuple, Literal
from pydantic import BaseModel, Field, model_validator

class Base64Codec:
    """bytes <-> base64(str) 변환 전담. urlsafe 지원 및 패딩 보정 포함."""

    @staticmethod
    def encode(data: bytes, *, urlsafe: bool = False) -> str:
        b = base64.urlsafe_b64encode(data) if urlsafe else base64.b64encode(data)
        return b.decode("ascii")

    @staticmethod
    def decode(s: str, *, urlsafe: bool = False) -> bytes:
        # 공백 제거 + 패딩 보정
        s = s.strip()
        m = len(s) % 4
        if m:
            s += "=" * (4 - m)
        try:
            return base64.urlsafe_b64decode(s) if urlsafe else base64.b64decode(s, validate=False)
        except Exception as e:
            raise ValueError(f"Invalid base64: {e}")

    @staticmethod
    def split_data_url(s: str) -> tuple[str, str]:
        """
        data URL이면 (mime, base64)로 분리, 아니면 ('application/octet-stream', s) 리턴
        예: data:image/png;base64,iVBORw0... → ("image/png", "iVBORw0...")
        """
        if s.startswith("data:"):
            header, b64 = s.split(",", 1)
            mime = "application/octet-stream"
            try:
                mime = header[5:].split(";")[0] or mime
            except Exception:
                pass
            return mime, b64
        return "application/octet-stream", s
    
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

class NDArrayB64(BaseModel):
    """
    JSON I/O 전용: buffer는 base64 문자열만 저장.
    urlsafe/dataURL도 지원(필드로 명시).
    """
    shape: Tuple[int, ...]
    dtype: str
    order: Literal["C", "F"] = "C"
    endian: Literal["Little", "Big"] = "Little"
    buffer_b64: str = Field(..., description="base64-encoded raw bytes")
    urlsafe: bool = False
    data_url: bool = False  # true면 buffer_b64가 data URL일 수 있음

    def __len__(self):
        return len(self.buffer_b64)
    
    @model_validator(mode="after")
    def _validate_buffer_and_size(self):
        # data URL이면 분리
        b64 = self.buffer_b64
        if self.data_url:
            _, b64 = Base64Codec.split_data_url(b64)

        raw = Base64Codec.decode(b64, urlsafe=self.urlsafe)

        # 예상 길이 검증(옵션이지만 조기 오류 발견에 유용)
        itemsize = np.dtype(self.dtype).itemsize
        expected = int(np.prod(self.shape)) * int(itemsize)
        if len(raw) != expected:
            raise ValueError(
                f"buffer size mismatch: expected {expected} bytes, got {len(raw)}"
            )
        return self

    # ========== 헬퍼 ==========
    @classmethod
    def from_ndarray(
        cls,
        arr: np.ndarray,
        *,
        order: Literal["C", "F"] = "C",
        endian: Literal["Little", "Big"] = "Little",
        urlsafe: bool = False,
        as_data_url: bool = False,
        mime: str = "application/octet-stream",
    ) -> "NDArrayB64":
        order_func = np.ascontiguousarray if order == "C" else np.asfortranarray
        arr = order_func(arr)
        desired = "<" if endian == "Little" else ">"
        arr = arr.astype(arr.dtype.newbyteorder(desired), copy=False)

        raw = arr.tobytes(order=order)
        b64 = Base64Codec.encode(raw, urlsafe=urlsafe)
        if as_data_url:
            b64 = f"data:{mime};base64,{b64}"

        return cls(
            shape=arr.shape,
            dtype=str(arr.dtype),
            order=order,
            endian=endian,
            buffer_b64=b64,
            urlsafe=urlsafe,
            data_url=as_data_url,
        )

    def to_ndarray(self) -> np.ndarray:
        b64 = self.buffer_b64
        if self.data_url:
            _, b64 = Base64Codec.split_data_url(b64)

        raw = Base64Codec.decode(b64, urlsafe=self.urlsafe)
        dt = np.dtype(self.dtype).newbyteorder("<" if self.endian == "Little" else ">")
        arr = np.frombuffer(raw, dtype=dt).reshape(self.shape, order=self.order)
        return np.ascontiguousarray(arr) if self.order == "C" else np.asfortranarray(arr)

    def to_internal(self) -> "NDArrayBuf":
        """내부 처리용 NDArrayBuf(bytes)로 변환"""
        b64 = self.buffer_b64
        if self.data_url:
            _, b64 = Base64Codec.split_data_url(b64)
        raw = Base64Codec.decode(b64, urlsafe=self.urlsafe)
        return NDArrayBuf(
            shape=self.shape,
            dtype=self.dtype,
            order=self.order,
            endian=self.endian,
            buffer=raw,
        )

def bench_mark(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    start = time.time()
    ret = func(*args, **kwargs)
    end = time.time()

    return ret, (end - start)