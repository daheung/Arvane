from abc import ABC, abstractmethod

class CallbackBase(ABC):
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
        self.batch_logs = None
        
    def on_batch_start(
        self, batch, logs=None
    ): pass

    def on_batch_end(
        self, batch, logs=None
    ): pass

    def on_epoch_start(
        self, epoch, logs=None
    ): pass

    def on_epoch_end(
        self, epoch, logs=None
    ): pass

    def on_predict_batch_start(
        self, batch, logs=None
    ): pass

    def on_predict_batch_end(
        self, batch, logs=None
    ): pass

    def on_predict_start(
        self, logs=None
    ): pass

    def on_predict_end(
        self, logs=None
    ): pass

    def on_test_batch_start(
        self, batch, logs=None
    ): pass

    def on_test_batch_end(
        self, batch, logs=None
    ): pass

    def on_test_start(
        self, logs=None
    ): pass

    def on_test_end(
        self, logs=None
    ): pass

    def on_train_batch_start(
        self, batch, logs=None
    ): pass

    def on_train_batch_end(
        self, batch, logs=None
    ): pass

    def on_train_start(
        self, logs=None
    ): pass

    def on_train_end(
        self, logs=None
    ): pass

    def log_dict(self, batch_size, sync_dict=True):
        pass