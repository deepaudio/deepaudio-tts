from omegaconf import DictConfig
import torch
from torch import Tensor, nn

from deepaudio.tts.models import register_model
from deepaudio.tts.models.base import BasePLModel
from deepaudio.tts.models.hifigan.hifigan import HiFiGANGenerator, HiFiGANMultiScaleMultiPeriodDiscriminator
from deepaudio.tts.models.hifigan.loss import FeatureMatchLoss, MelSpectrogramLoss, GeneratorAdversarialLoss, \
    DiscriminatorAdversarialLoss

from .configurations import HifiganConfigs


@register_model('hifigan', dataclass=HifiganConfigs)
class HifiganModel(BasePLModel):
    def __init__(self, configs: DictConfig):
        super(HifiganModel, self).__init__(configs)

    def build_model(self):
        self.generator = HiFiGANGenerator(
            in_channels=self.configs.model.in_channels,
            out_channels=self.configs.model.out_channels,
            channels=self.configs.model.channels,
            global_channels=self.configs.model.global_channels,
            kernel_size=self.configs.model.kernel_size,
            upsample_scales=self.configs.model.upsample_scales,
            upsample_kernel_sizes=self.configs.model.upsample_kernel_sizes,
            resblock_kernel_sizes=self.configs.model.resblock_kernel_sizes,
            resblock_dilations=self.configs.model.resblock_dilations,
            use_additional_convs=self.configs.model.use_additional_convs,
            bias=self.configs.model.bias,
            nonlinear_activation=self.configs.model.nonlinear_activation,
            nonlinear_activation_params=self.configs.model.nonlinear_activation_params,
            use_weight_norm=self.configs.model.use_weight_norm
        )
        self.discriminator = HiFiGANMultiScaleMultiPeriodDiscriminator(
            scales=self.configs.model.scales,
            scale_downsample_pooling=self.configs.model.scale_downsample_pooling,
            scale_downsample_pooling_params=self.configs.model.scale_downsample_pooling_params,
            scale_discriminator_params=self.configs.model.scale_discriminator_params,
            follow_official_norm=self.configs.model.follow_official_norm,
            periods=self.configs.model.periods,
            period_discriminator_params=self.configs.model.period_discriminator_params,
        )

    def configure_criterion(self) -> nn.Module:
        self.criterion_feat_match = FeatureMatchLoss()
        self.criterion_mel = MelSpectrogramLoss()
        self.criterion_gen_adv = GeneratorAdversarialLoss()
        self.criterion_dis_adv = DiscriminatorAdversarialLoss()


    def training_step(self, batch: tuple, batch_idx: int, optimizer_idx: int):
        losses_dict = {}
        # parse batch
        wav, mel = batch

        # Generator
        if self.state.iteration > self.generator_train_start_steps:
            # (B, out_channels, T ** prod(upsample_scales)
            wav_ = self.generator(mel)

            # initialize
            gen_loss = 0.0
            aux_loss = 0.0

            # mel spectrogram loss
            mel_loss = self.criterion_mel(wav_, wav)
            aux_loss += mel_loss
            losses_dict["mel_loss"] = mel_loss

            gen_loss += aux_loss * self.lambda_aux

            # adversarial loss
            if self.state.iteration > self.discriminator_train_start_steps:
                p_ = self.discriminator(wav_)
                adv_loss = self.criterion_gen_adv(p_)
                losses_dict["adversarial_loss"] = adv_loss

                # feature matching loss
                # no need to track gradients
                with torch.no_grad():
                    p = self.discriminator(wav)
                fm_loss = self.criterion_feat_match(p_, p)
                losses_dict["feature_matching_loss"] = float(fm_loss)

                adv_loss += self.lambda_feat_match * fm_loss

                gen_loss += self.lambda_adv * adv_loss

            losses_dict["generator_loss"] = float(gen_loss)


        # Disctiminator
        if self.state.iteration > self.discriminator_train_start_steps:
            # re-compute wav_ which leads better quality
            with torch.no_grad():
                wav_ = self.generator(mel)

            p = self.discriminator(wav)
            p_ = self.discriminator(wav_.detach())
            real_loss, fake_loss = self.criterion_dis_adv(p_, p)
            dis_loss = real_loss + fake_loss
            losses_dict["real_loss"] = real_loss
            losses_dict["fake_loss"] = fake_loss
            losses_dict["discriminator_loss"] = dis_loss
