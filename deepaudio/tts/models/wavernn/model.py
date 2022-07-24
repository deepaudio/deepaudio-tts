from omegaconf import DictConfig
import torch
from torch import Tensor, nn

from deepaudio.tts.models import register_model
from deepaudio.tts.models.base import BasePLModel
from deepaudio.tts.models.wavernn import WaveRNN
from deepaudio.tts.modules.losses import discretized_mix_logistic_loss

from .configurations import WaveRNNConfigs


@register_model('wavernn', dataclass=WaveRNNConfigs)
class WaveRNNModel(BasePLModel):
    def __init__(self, configs: DictConfig):
        super(WaveRNNModel, self).__init__(configs)

    def build_model(self):
        self.model = WaveRNN()

    def configure_criterion(self):
        if self.configs.model.mode == 'RAW':
            self.criterion = nn.CrossEntropyLoss()
        elif self.configs.model.mode == 'MOL':
            self.criterion = discretized_mix_logistic_loss()
        else:
            self.criterion = None
            RuntimeError('Unknown model mode value - ', self.configs.model.mode)

    def compute_loss(self, batch):
        wav, y, mel = batch
        y_hat = self.model(wav, mel)
        if self.mode == 'RAW':
            y_hat = y_hat.transpose([0, 2, 1]).unsqueeze(-1)
        elif self.mode == 'MOL':
            y_hat = y_hat.type(torch.float32)

        y = y.unsqueeze(-1)
        loss = self.criterion(y_hat, y)
        return loss

    def training_step(self, batch: tuple, batch_idx: int):
        loss = self.compute_loss(batch)
        return {
            'loss': loss
        }

    def validation_step(self, batch: tuple, batch_idx: int):
        loss = self.compute_loss(batch)
        return {
            'val_loss': loss
        }
