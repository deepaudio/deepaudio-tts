from omegaconf import DictConfig
import torch
from torch import Tensor, nn

from deepaudio.tts.models import register_model
from deepaudio.tts.models.base import BasePLModel
from deepaudio.tts.models.melgan.melgan import MelGANGenerator, MelGANMultiScaleDiscriminator
from deepaudio.tts.models.melgan.style_melgan import StyleMelGANDiscriminator, StyleMelGANGenerator

from deepaudio.tts.modules.losses import DiscriminatorAdversarialLoss
from deepaudio.tts.modules.losses import GeneratorAdversarialLoss
from deepaudio.tts.modules.losses import MultiResolutionSTFTLoss
from deepaudio.tts.models.melgan.pqmf import PQMF
from .configurations import MelGANConfigs, StyleMelGANConfigs


@register_model('melgan', dataclass=MelGANConfigs)
class MelganModel(BasePLModel):
    def __init__(self, configs: DictConfig):
        super(MelganModel, self).__init__(configs)

    def build_model(self):
        self.generator = MelGANGenerator(
            in_channels=self.configs.model.in_channels,
            out_channels=self.configs.model.out_channels,
            kernel_size=self.configs.model.kernel_size,
            channels=self.configs.model.channels,
            bias=self.configs.model.bias,
            upsample_scales=self.configs.model.upsample_scales,
            stack_kernel_size=self.configs.model.stack_kernel_size,
            stacks=self.configs.model.stacks,
            nonlinear_activation=self.configs.model.nonlinear_activation,
            nonlinear_activation_params=self.configs.model.nonlinear_activation_params,
            pad=self.configs.model.pad,
            pad_params=self.configs.model.pad_params,
            use_final_nonlinear_activation=self.configs.model.use_final_nonlinear_activation,
            use_weight_norm=self.configs.model.use_weight_norm,
        )
        self.discriminator = MelGANMultiScaleDiscriminator(
            in_channels=self.configs.model.in_channels_discriminator,
            out_channels=self.configs.model.out_channels_discriminator,
            scales=self.configs.model.scales_discriminator,
            downsample_pooling=self.configs.model.downsample_pooling_discriminator,
            downsample_pooling_params=self.configs.model.downsample_pooling_params_discriminator,
            kernel_sizes=self.configs.model.kernel_sizes_discriminator,
            channels=self.configs.model.channels_discriminator,
            max_downsample_channels=self.configs.model.max_downsample_channels_discriminator,
            bias=self.configs.model.bias_discriminator,
            downsample_scales=self.configs.model.downsample_scales_discriminator,
            nonlinear_activation=self.configs.model.nonlinear_activation_discriminator,
            nonlinear_activation_params=self.configs.model.nonlinear_activation_params_discriminator,
            pad=self.configs.model.pad_discriminator,
            pad_params=self.configs.model.pad_params_discriminator,
            use_weight_norm=self.configs.model.use_weight_norm_discriminator,
        )

    def configure_criterion(self) -> nn.Module:
        self.criterion_stft = MultiResolutionSTFTLoss(**self.configs.model.stft_loss_params)
        self.criterion_sub_stft = MultiResolutionSTFTLoss(
            **self.configs.model.subband_stft_loss_params)
        self.criterion_gen_adv = GeneratorAdversarialLoss()
        self.criterion_dis_adv = DiscriminatorAdversarialLoss()
        self.criterion_pqmf = PQMF(subbands=self.configs.out_channels_discriminator)

    def training_step(self, batch: tuple, batch_idx: int, optimizer_idx: int):
        losses_dict = {}
        # parse batch
        wav, mel = batch
        # Generator
        if optimizer_idx == 0:
            # (B, out_channels, T ** prod(upsample_scales)

            wav_ = self.generator(mel)
            wav_mb_ = wav_
            # (B, 1, out_channels*T ** prod(upsample_scales)
            wav_ = self.criterion_pqmf.synthesis(wav_mb_)

            # initialize
            gen_loss = 0.0
            aux_loss = 0.0

            # full band Multi-resolution stft loss
            sc_loss, mag_loss = self.criterion_stft(wav_, wav)
            # for balancing with subband stft loss
            # Eq.(9) in paper
            aux_loss += 0.5 * (sc_loss + mag_loss)
            losses_dict["spectral_convergence_loss"] = sc_loss
            losses_dict["log_stft_magnitude_loss"] = mag_loss

            # sub band Multi-resolution stft loss
            # (B, subbands, T // subbands)
            wav_mb = self.criterion_pqmf.analysis(wav)
            sub_sc_loss, sub_mag_loss = self.criterion_sub_stft(wav_mb_, wav_mb)
            # Eq.(9) in paper
            aux_loss += 0.5 * (sub_sc_loss + sub_mag_loss)
            losses_dict["sub_spectral_convergence_loss"] = sub_sc_loss
            losses_dict["sub_log_stft_magnitude_loss"] = sub_mag_loss

            gen_loss += aux_loss * self.lambda_aux

            # adversarial loss TODO
            p_ = self.discriminator(wav_)
            adv_loss = self.criterion_gen_adv(p_)
            losses_dict["adversarial_loss"] = float(adv_loss)

            gen_loss += self.lambda_adv * adv_loss


        # Disctiminator
        if optimizer_idx == 1:
            # re-compute wav_ which leads better quality
            with torch.no_grad():
                wav_ = self.generator(mel)
            wav_ = self.criterion_pqmf.synthesis(wav_)
            p = self.discriminator(wav)
            p_ = self.discriminator(wav_.detach())
            real_loss, fake_loss = self.criterion_dis_adv(p_, p)
            dis_loss = real_loss + fake_loss
            losses_dict["real_loss"] = real_loss
            losses_dict["fake_loss"] = fake_loss
            losses_dict["discriminator_loss"] = dis_loss



