import os
import sys
import yaml
import box
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def load_config():        
    run_mode = os.getenv("MODE", "development")
    config_path = os.path.join(os.getcwd(), "Arvane/config")

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

    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        config.accelerator = "gpu"
        config.n_devices = n_gpus
    else:
        config.accelerator = "cpu"
        config.n_devices = 1
        
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config