import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Optional
import psutil
"""
TODO: 
0. log model infos into run file on init start
1. CKPT auto-save
2. testing
3. prediction: 2D case visualization

"""

class MemoryMonitoringCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.process = psutil.Process()

    def log_memory_usage(self, stage):
        # 获取当前 Python 进程的内存使用情况
        memory_info = self.process.memory_info()
        memory_used = memory_info.rss / (1024 ** 3)  # 转换为 GB

        # 获取 GPU 内存信息（如果有 GPU）
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # 转换为 GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # 转换为 GB
        else:
            gpu_memory_allocated = 0
            gpu_memory_reserved = 0

        print(f"[{stage}] Python Process Memory: {memory_used:.2f} GB used")
        if torch.cuda.is_available():
            print(f"[{stage}] GPU Memory: {gpu_memory_allocated:.2f} GB allocated / {gpu_memory_reserved:.2f} GB reserved")

    def on_train_epoch_start(self, trainer, pl_module):
        self.log_memory_usage('Train Start')

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_memory_usage('Train End')

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_memory_usage('Validation Start')

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_memory_usage('Validation End')

    def on_test_epoch_start(self, trainer, pl_module):
        self.log_memory_usage('Test Start')

    def on_test_epoch_end(self, trainer, pl_module):
        self.log_memory_usage('Test End')

# class LossNomorlizationCallback(L.Callback):
#     def __init__(self, std:Optional[float]=None) -> None:
#         super().__init__()
#         self.std = std
#         epsilon = 1e-6
#         if self.std is not None:
#             self.coeff = 1 / (self.std**2 + epsilon)
#         else:
#             self.coeff = 1

#     def on_before_backward(self, trainer: L.Trainer, pl_module: L.LightningModule, loss: torch.Tensor) -> None:
#         return super().on_before_backward(trainer, pl_module, loss*self.coeff)

