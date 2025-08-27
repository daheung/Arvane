import os
import box
import yaml
import torch

def load_config(config_fname):
    with open(config_fname, "r") as f:
        config = box.Box(yaml.safe_load(f))

    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        config.accelerator = "gpu"
        config.n_devices = n_gpus
    else:
        config.accelerator = "cpu"
        config.n_devices = 1

    return config