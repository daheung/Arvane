import argparse
import os
import sys
import shutil

import box
import pytorch_lightning as pl
import torch
import yaml

if os.path.join(os.path.abspath(os.curdir)) not in sys.path:
    sys.path.append(os.path.join(os.path.abspath(os.curdir)))

import source.reconstruction.data as data
import source.reconstruction.fine_recon as fine_recon
import source.reconstruction.fine_recon_callback as fine_recon_callback
import source.reconstruction.utils as utils

torch.set_float32_matmul_precision('medium')

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

@pl.utilities.rank_zero_only
def zip_code(save_dir):
    os.system(f"zip {save_dir}/code.zip *.py config.yml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/config.yml")
    parser.add_argument("--task", default="predict")
    parser.add_argument("--ckpt", default="./checkpoints/fine_recon.ckpt")
    args = parser.parse_args()

    if args.ckpt is not None:
        shutil.copy(args.ckpt, args.ckpt + ".bak")

    config = load_config(args.config)
    if args.task == "predict":
        config.n_devices = 1

    model = fine_recon.FineRecon(config)

    logger = pl.loggers.TensorBoardLogger(save_dir=".", version=config.run_name)
    logger.experiment

    # zip_code(logger.experiment.log_dir)

    trainer = pl.Trainer(
        logger=logger,
        accelerator=config.accelerator,
        devices=config.n_devices,
        max_steps=config.steps,
        log_every_n_steps=50,
        precision=16,
        strategy="ddp" if config.n_devices > 1 else "auto",
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="loss_val/loss", save_top_k=10),
            fine_recon_callback.FineReconCallback(config)
        ],
    )

    trainer.predict(model, ckpt_path=args.ckpt)
    
