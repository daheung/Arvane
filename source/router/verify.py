from enum import Enum
from typing import Any

def check_dtype(dtype: str) -> bool:
    if (dtype in ['float8', 'float16', 'float32', 'float64']):
        return True
    
    return False

class InvalidRequestQueryType(Enum):
    IQT_Dtype = "dtype"
    IQT_Image = "image"
    
class InvalidRequestQueryException(Exception):
    def __init__(self, type: InvalidRequestQueryType, current: Any):
        super().__init__(f"Invalid key in request query: {type.value}. current: {current}.")

class InvalidRequestContentTypeException(Exception):
    def __init__(self, current: Any):
        super().__init__(f"Invalid content type. current: {current}.")

class InvalidRequestBodyException(Exception):
    def __init__(self, current: Any):
        super().__init__(f"Invalid body payload. current: {current}")