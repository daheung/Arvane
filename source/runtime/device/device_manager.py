import sys
import torch
import psutil
import pynvml
import logging

from enum import IntEnum
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
class DeviceMemoryDescription:
    total_memory: int
    free_memory: int
    used_memory: int

@dataclass
class DeviceVideoMemoryDescription(DeviceMemoryDescription):
    ...

@dataclass
class DeviceVirtualMemoryDescription(DeviceMemoryDescription):
    ...

@dataclass
class DeviceDescriptor:
    device_index: int
    device: torch.device
    description: str
    memory: DeviceMemoryDescription

# @dataclass
class DeviceManager:
    def __init__(self, enable_cpu: bool = False):
        self.num_device = torch.cuda.device_count()
        self.cpu_device_desc: Optional[DeviceDescriptor] = None
        self.gpu_device_desc: List[DeviceDescriptor] = []
        self.enable_cpu = enable_cpu

        logging.info(f"DeviceManager initialized. GPU count: {self.num_device}, CPU enabled: {self.enable_cpu}")

        if (self.enable_cpu):
            descriptor: DeviceDescriptor = self._build_cpu_descriptor()
            self.cpu_device_desc = descriptor

            logging.info("CPU device detected.")
            logging.info("  Device name: CPU Device")
            logging.info("  Device type: CPU")
            logging.info("  Virtual memory info:")
            logging.info(f"    total: {descriptor.memory.total_memory // (1024 ** 2)} MiB")
            logging.info(f"    free : {descriptor.memory.free_memory // (1024 ** 2)} MiB")
            logging.info(f"    used : {descriptor.memory.used_memory // (1024 ** 2)} MiB")

        for idx in range(self.num_device):
            descriptor: DeviceDescriptor = self._build_descriptor(idx)
            self.gpu_device_desc.append(descriptor)

            total_memory_mib: int = int(descriptor.memory.total_memory / (1024 ** 2))
            free_memory_mib: int = int(descriptor.memory.free_memory / (1024 ** 2))
            used_memory_mib: int = int(descriptor.memory.used_memory / (1024 ** 2))
            logging.info(f'GPU:{idx} detected.')
            logging.info(f'  GPU name: {descriptor.description}')
            logging.info(f'  Cuda GPU str: {descriptor.device}')
            logging.info(f'  Dedicated video memory info:')
            logging.info(f'    total: {total_memory_mib} MiB')
            logging.info(f'    free : {free_memory_mib} MiB')
            logging.info(f'    used : {used_memory_mib} MiB')
    
    def gpu_desc(self, idx: int) -> Optional[DeviceDescriptor]:
        if (len(self.gpu_device_desc) < idx + 1):
            return None
        
        self.gpu_device_desc[idx] = self._build_descriptor(idx)
        return self.gpu_device_desc[idx]
    
    def gpu_num(self) -> int:
        return self.num_device
    
    def get_device_considering_slack(self, required_minimum_memory_mib: Optional[int] = None) -> Optional[DeviceDescriptor]:
        best_descriptor: Optional[DeviceDescriptor] = None
        for idx in range(self.num_device):
            gpu_descriptor: DeviceDescriptor = self._build_descriptor(idx)
            if (best_descriptor is None):
                best_descriptor = gpu_descriptor

            free_memory = best_descriptor.memory.free_memory
            if (free_memory < gpu_descriptor.memory.free_memory):
                best_descriptor = gpu_descriptor
        
        if (self.enable_cpu and required_minimum_memory_mib is not None):
            cpu_descriptor: DeviceDescriptor = self._build_cpu_descriptor()
            free_memory_mib = best_descriptor.memory.free_memory // (1024 ** 2)
            if (free_memory_mib < required_minimum_memory_mib):
                logging.warning(f"All GPU devices have less than required minimum memory {required_minimum_memory_mib} MiB. Falling back to CPU device.")
                best_descriptor = cpu_descriptor

        return best_descriptor
    
    def _build_cpu_descriptor(self) -> DeviceDescriptor:
        vm = psutil.virtual_memory()

        total_memory = vm.total
        free_memory  = vm.available
        used_memory  = total_memory - free_memory

        return DeviceDescriptor(
            device=torch.device("cpu"),
            device_index=-1,
            description="CPU Device",
            memory=DeviceVideoMemoryDescription(
                total_memory=total_memory,
                free_memory=free_memory,
                used_memory=used_memory,
            ),
        )

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
            memory=DeviceVideoMemoryDescription(
                total_memory=mem_info.total,
                free_memory=mem_info.free,
                used_memory=mem_info.used
            ),
        )