import pytorch_lightning as pl

class FineReconCallback(pl.Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.run_name = config.run_name
        self.ckpt_path = config.ckpt
        self.outputs = list()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.outputs.append(outputs)