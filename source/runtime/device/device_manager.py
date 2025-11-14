import sys
import torch
import pynvml
import logging
from typing import Tuple, List, Optional
from dataclasses import dataclass

pynvml.nvmlInit()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

@dataclass
class DeviceVideoMemoryDescription:
    total_memory: int
    free_memory: int
    used_memory: int

@dataclass
class DeviceDescriptor:
    device_index: int
    device: torch.device
    description: str
    dedicated_video_memory: DeviceVideoMemoryDescription

class DeviceManager:
    def __init__(self):
        self.num_gpus = torch.cuda.device_count()
        self.device_desc: List[DeviceDescriptor] = []
        
        for idx in range(self.num_gpus):
            descriptor: DeviceDescriptor = self._build_descriptor(idx)
            self.device_desc.append(descriptor)

            total_memory_per_mib: int = int(descriptor.dedicated_video_memory.total_memory / (1024 ** 2))
            free_memory_per_mib: int = int(descriptor.dedicated_video_memory.free_memory / (1024 ** 2))
            used_memory_per_mib: int = int(descriptor.dedicated_video_memory.used_memory / (1024 ** 2))

            logging.info(f'GPU:{idx} detected.')
            logging.info(f'  GPU name: {descriptor.description}')
            logging.info(f'  Cuda GPU str: {descriptor.device}')
            logging.info(f'  Dedicated video memory info:')
            logging.info(f'    total: {total_memory_per_mib} MiB')
            logging.info(f'    free : {free_memory_per_mib} MiB')
            logging.info(f'    used : {used_memory_per_mib} MiB')
    
    def gpu_desc(self, idx: int) -> Optional[DeviceDescriptor]:
        if (len(self.device_desc) < idx + 1):
            return None
        
        self.device_desc[idx] = self._build_descriptor(idx)
        return self.device_desc[idx]
    
    def gpu_num(self) -> int:
        return self.num_gpus
    
    def _gpu_update(self):
        for idx in range(self.num_gpus):
            self.device_desc[idx] = self._build_descriptor(idx)

    def get_gpu_considering_slack(self) -> Optional[Tuple[int, DeviceDescriptor]]:
        best_descriptor: Optional[Tuple[int, DeviceDescriptor]] = None
        for idx in range(self.num_gpus):
            descriptor: DeviceDescriptor = self._build_descriptor(idx)
            if (best_descriptor is None):
                best_descriptor = (idx, descriptor)

            free_memory = best_descriptor[1].dedicated_video_memory.free_memory
            if (free_memory < descriptor.dedicated_video_memory.free_memory):
                best_descriptor = (idx, descriptor)
        
        return best_descriptor
    
    def _build_descriptor(self, idx: int) -> DeviceDescriptor:
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        name = pynvml.nvmlDeviceGetName(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        if isinstance(name, bytes):
            name = name.decode("utf-8")
        
        return DeviceDescriptor(
            device=torch.device(f"cuda:{idx}"),
            device_index=idx,
            description=str(name),
            dedicated_video_memory=DeviceVideoMemoryDescription(
                total_memory=mem_info.total,
                free_memory=mem_info.free,
                used_memory=mem_info.used
            ),
        )