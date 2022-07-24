from omegaconf import DictConfig
import torch
from torch import Tensor, nn

from deepaudio.tts.models import register_model
from deepaudio.tts.models.base import BasePLModel
from deepaudio.tts.models.parallel_wavegan import ParallelWaveGANDiscriminator
from deepaudio.tts.models.parallel_wavegan import ParallelWaveGANGenerator


from deepaudio.tts.modules.losses import MultiResolutionSTFTLoss
from deepaudio.tts.models.melgan.pqmf import PQMF
from .configurations import ParallelWaveganConfigs


@register_model('parallel_wavegan', dataclass=ParallelWaveganConfigs)
class MelganModel(BasePLModel):
    def __init__(self, configs: DictConfig):
        super(MelganModel, self).__init__(configs)

    def build_model(self):
        self.generator = ParallelWaveGANGenerator(
            in_channels=self.configs.model.in_channel,
            out_channels=self.configs.model.out_channel,
            kernel_size=self.configs.model.kernel_size,
            layers=self.configs.model.layers,
            stacks=self.configs.model.stacks,
            residual_channels=self.configs.model.residual_channels,
            gate_channels=self.configs.model.gate_channels,
            skip_channels=self.configs.model.skip_channels,
            aux_channels=self.configs.model.aux_channels,
            aux_context_window=self.configs.model.aux_context_window,
            dropout_rate=self.configs.model.dropout_rate,
            bias=self.configs.model.bias,
            use_weight_norm=self.configs.model.use_weight_norm,
            upsample_conditional_features=self.configs.model.upsample_conditional_features,
            upsample_net=self.configs.model.upsample_net,
            upsample_params=self.configs.model.upsample_params
        )
        self.discriminator = ParallelWaveGANDiscriminator(
            in_channels=self.configs.model.in_channels_discriminator,
            out_channels=self.configs.model.out_channels_discriminator,
            kernel_size=self.configs.model.kernel_size_discriminator,
            layers=self.configs.model.layers_discriminator,
            conv_channels=self.configs.model.conv_channels_discriminator,
            dilation_factor=self.configs.model.dilation_factor_discriminator,
            nonlinear_activation=self.configs.model.nonlinear_activation_discriminator,
            nonlinear_activation_params=self.configs.model.nonlinear_activation_params_discriminator,
            bias=self.configs.model.bias_discriminator,
            use_weight_norm=self.configs.model.use_weight_norm_discriminator,
        )

    def configure_criterion(self) -> nn.Module:
        self.criterion_stft = MultiResolutionSTFTLoss(self.configs.model.stft_loss_params)
        self.criterion_mse = nn.MSELoss()

    def training_step(self, batch: tuple, batch_idx: int, optimizer_idx: int):
        losses_dict = {}
        # parse batch
        wav, mel = batch

        # Generator
        if self.state.iteration > self.generator_train_start_steps:
            noise = torch.randn(wav.shape)
            wav_ = self.generator(noise, mel)

            # initialize
            gen_loss = 0.0
            aux_loss = 0.0

            # multi-resolution stft loss
            sc_loss, mag_loss = self.criterion_stft(wav_, wav)
            aux_loss += sc_loss + mag_loss

            gen_loss += aux_loss * self.lambda_aux

            losses_dict["spectral_convergence_loss"] = sc_loss
            losses_dict["log_stft_magnitude_loss"] = mag_loss

            # adversarial loss
            if self.state.iteration > self.discriminator_train_start_steps:
                p_ = self.discriminator(wav_)
                adv_loss = self.criterion_mse(p_, torch.ones_like(p_))
                losses_dict["adversarial_loss"] = adv_loss

                gen_loss += self.configs.model.lambda_adv * adv_loss

            losses_dict["generator_loss"] = gen_loss


        # Disctiminator
        if self.state.iteration > self.discriminator_train_start_steps:
            with torch.no_grad():
                wav_ = self.generator(noise, mel)
            p = self.discriminator(wav)
            p_ = self.discriminator(wav_.detach())
            real_loss = self.criterion_mse(p, torch.ones_like(p))
            fake_loss = self.criterion_mse(p_, torch.zeros_like(p_))
            dis_loss = real_loss + fake_loss

            losses_dict["real_loss"] = float(real_loss)
            losses_dict["fake_loss"] = float(fake_loss)
            losses_dict["discriminator_loss"] = float(dis_loss)

