from omegaconf import DictConfig
from torch import Tensor, nn

from deepaudio.tts.models import register_model
from deepaudio.tts.models.base import BasePLModel
from deepaudio.tts.models.vits import VITS
from deepaudio.tts.modules.losses import MelSpectrogramLoss
from deepaudio.tts.modules.losses import FeatureMatchLoss
from deepaudio.tts.modules.losses import GeneratorAdversarialLoss
from deepaudio.tts.modules.losses import DiscriminatorAdversarialLoss

from deepaudio.tts.models.vits.loss import KLDivergenceLoss

from .configurations import VitsConfigs


@register_model('vits', dataclass=VitsConfigs)
class VitsModel(BasePLModel):
    def __init__(self, configs: DictConfig):
        super(VitsModel, self).__init__(configs)

    def build_model(self):
        self.model = VITS()

    def configure_criterion(self) -> nn.Module:
        self.criterion_mel = MelSpectrogramLoss(
            **config["mel_loss_params"], )
        self.criterion_feat_match = FeatureMatchLoss(
            **config["feat_match_loss_params"], )
        self.criterion_gen_adv = GeneratorAdversarialLoss(
            **config["generator_adv_loss_params"], )
        self.criterion_dis_adv = DiscriminatorAdversarialLoss(
            **config["discriminator_adv_loss_params"], )
        self.criterion_kl = KLDivergenceLoss()

