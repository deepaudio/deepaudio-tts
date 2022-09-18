from omegaconf import DictConfig
import torch
from torch import Tensor, nn

from pytorch_lightning import LightningModule
from deepaudio.tts.models.parallel_wavegan import ParallelWaveGANDiscriminator
from deepaudio.tts.models.parallel_wavegan import ParallelWaveGANGenerator


from deepaudio.tts.modules.losses import MultiResolutionSTFTLoss


class MelganModel(LightningModule):
    def __init__(self,
                 generator: ParallelWaveGANGenerator,
                 discriminator: ParallelWaveGANDiscriminator,
                 criterion_stft: MultiResolutionSTFTLoss,
                 criterion_mse: torch.nn.MSELoss,
                 optimizer_d: torch.optim.Optimizer,
                 scheduler_d: torch.optim.lr_scheduler,
                 optimizer_g: torch.optim.Optimizer,
                 scheduler_g: torch.optim.lr_scheduler,
                 ):
        super(MelganModel, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.criterion_stft = criterion_stft
        self.criterion_mse = criterion_mse

    def training_step(self, batch: tuple, batch_idx: int, optimizer_idx: int):
        losses_dict = {}
        # parse batch
        wav, mel = batch

        # Generator
        if optimizer_idx == 0:
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
        if optimizer_idx == 1:
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

    def configure_optimizers(self):
        optimizer_g = self.hparams.optimizer_g(params=self.generator.parameters())
        optimizer_d = self.hparams.optimizer_d(params=self.discriminator.parameters())
        scheduler_g = self.hparams.scheduler_g(optimizer=optimizer_g)
        scheduler_d = self.hparams.scheduler_d(optimizer=optimizer_d)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]



