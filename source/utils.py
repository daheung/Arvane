import os
import sys
import yaml
import box
import torch
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def load_config():        
    run_mode = os.getenv("MODE", "development")
    config_path = os.path.join(os.getcwd(), "config")

    if run_mode == "development":
        depth_config = os.path.join(config_path, "depth-dev.yml")
        recon_config = os.path.join(config_path, "recon-dev.yml")
    else:
        depth_config = os.path.join(config_path, "depth-prod.yml")
        recon_config = os.path.join(config_path, "recon-prod.yml")

    logging.info(f"Depth configuration PATH: {depth_config}")
    logging.info(f"Reconstruction configuration PATH: {recon_config}")
    return load_config_impl(depth_config), load_config_impl(recon_config)

def load_config_impl(config_fname):
    with open(config_fname, "r") as f:
        data = yaml.safe_load(f) or {}
        config = box.Box(data)

    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        config.accelerator = "gpu"
        config.n_devices = n_gpus
    else:
        config.accelerator = "cpu"
        config.n_devices = 1
        
    return config

def np_to_torch_dtype(dt: np.dtype) -> torch.dtype:
    dt = np.dtype(dt)  # 문자열/타입도 허용
    table = {
        np.dtype('bool')      : torch.bool,
        np.dtype('uint8')     : torch.uint8,
        np.dtype('int8')      : torch.int8,
        np.dtype('int16')     : torch.int16,
        np.dtype('int32')     : torch.int32,
        np.dtype('int64')     : torch.int64,
        np.dtype('float16')   : torch.float16,
        np.dtype('float32')   : torch.float32,
        np.dtype('float64')   : torch.float64,
        np.dtype('complex64') : torch.complex64,
        np.dtype('complex128'): torch.complex128,
    }
    if dt.name == 'bfloat16':
        return torch.bfloat16
    if dt in table:
        return table[dt]
    raise TypeError(f"Unsupported NumPy dtype for PyTorch: {dt}")

def torch_to_np_dtype(dt: torch.dtype) -> np.dtype:
    table = {
        torch.bool       : np.bool_,
        torch.uint8      : np.uint8,
        torch.int8       : np.int8,
        torch.int16      : np.int16,
        torch.int32      : np.int32,
        torch.int64      : np.int64,
        torch.float16    : np.float16,
        torch.float32    : np.float32,
        torch.float64    : np.float64,
        torch.complex64  : np.complex64,
        torch.complex128 : np.complex128,
    }
    if dt is torch.bfloat16:
        # 대부분의 NumPy 빌드에선 bfloat16 미지원
        # np.dtype('bfloat16')가 없다면 직접 생성 불가
        try:
            return np.dtype('bfloat16')
        except TypeError:
            raise TypeError("NumPy has no native bfloat16 on this platform.")
    if dt in table:
        return np.dtype(table[dt])
    raise TypeError(f"Unsupported torch dtype for NumPy: {dt}")