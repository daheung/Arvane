import time
from enum import Enum

DepthObjectArray_Config = {
    "chunk_size": 8,
}

class DepthObjectArray:
    def __init__(self):
        self.object_array = list

    def add_object(self, depth):
        tid = time.time()
        internal_id = (len(self.object_array))

        self.object_array.append({
            "t-id": tid,
            "internal-id": internal_id,
            "depth-object": depth
        })
    
    @property
    def raw_object(self, idx):
        return self.object(idx)['depth-object']
    
    @property
    def object(self, idx):
        if (self._check_idx(idx)):
            return None
        
        return self.object_array[idx]
    
    @property
    def chunk(self, idx, chunk_size):
        if (self._check_idx(idx + chunk_size)):
            return None
        
        return self.object_array[idx:chunk_size]

    def _check_idx(self, idx):
        return (self.object_array) and len(self.object_array) and (len(self.object_array) is not idx)
    
        
class InferObjectArray:
    def __init__(self, config):
        self.depth_array = DepthObjectArray()
        self.config = config
    
    @property
    def object(self, idx):
        return self.depth_array