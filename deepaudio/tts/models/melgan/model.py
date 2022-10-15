import torch
from torch import Tensor, nn

from pytorch_lightning import LightningModule
from deepaudio.tts.models.melgan import MelGANGenerator, MelGANMultiScaleDiscriminator
from deepaudio.tts.models.melgan.style_melgan import StyleMelGANDiscriminator, StyleMelGANGenerator

from deepaudio.tts.modules.losses import DiscriminatorAdversarialLoss
from deepaudio.tts.modules.losses import GeneratorAdversarialLoss
from deepaudio.tts.modules.losses import MultiResolutionSTFTLoss
from deepaudio.tts.models.melgan.pqmf import PQMF


class MelganModel(LightningModule):
    def __init__(self,
                 generator: MelGANGenerator,
                 discriminator: MelGANMultiScaleDiscriminator,
                 criterion_stft: MultiResolutionSTFTLoss,
                 criterion_sub_stft: MultiResolutionSTFTLoss,
                 criterion_gen_adv: GeneratorAdversarialLoss,
                 criterion_dis_adv: DiscriminatorAdversarialLoss,
                 criterion_pqmf: PQMF,
                 lambda_aux: float,
                 lambda_adv: float,
                 optimizer_d: torch.optim.Optimizer,
                 scheduler_d: torch.optim.lr_scheduler,
                 optimizer_g: torch.optim.Optimizer,
                 scheduler_g: torch.optim.lr_scheduler,
                 discriminator_train_start_steps: int = 100000,
                 ):
        super(MelganModel, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.criterion_stft = criterion_stft
        self.criterion_sub_stft = criterion_sub_stft
        self.criterion_gen_adv = criterion_gen_adv
        self.criterion_dis_adv = criterion_dis_adv
        self.criterion_pqmf = criterion_pqmf
        self.save_hyperparameters(logger=False, ignore=["generator",
                                                        "discriminator",
                                                        "criterion_stft",
                                                        "criterion_sub_stft",
                                                        "criterion_gen_adv",
                                                        "criterion_dis_adv",
                                                        "criterion_pqmf",])
        self.automatic_optimization = False

    def step_generator(self, wav, mel, batch_idx):
        losses_dict = {}
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

        gen_loss += aux_loss * self.hparams.lambda_aux

        # adversarial loss
        if batch_idx > self.hparams.discriminator_train_start_steps:
            p_ = self.discriminator(wav_)
            adv_loss = self.criterion_gen_adv(p_)
            losses_dict["adversarial_loss"] = float(adv_loss)
            gen_loss += self.hparams.lambda_adv * adv_loss
        self.log_dict(losses_dict)
        return gen_loss

    def step_disctiminator(self, wav, mel):
        losses_dict = {}
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
        self.log_dict(losses_dict)
        return dis_loss

    def training_step(self, batch: tuple, batch_idx: int, optimizer_idx: int):
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()

        # parse batch
        wav, mel = batch
        # Generator
        gen_loss = self.step_generator(wav, mel, batch_idx)
        opt_g.zero_grad()
        self.manual_backward(gen_loss)
        opt_g.step()
        sch_g.step()

        # Disctiminator
        if batch_idx > self.hparams.discriminator_train_start_steps:
            #re-compute wav_ which leads better quality
            dis_loss = self.step_disctiminator(wav, mel)
            opt_d.zero_grad()
            self.manual_backward(dis_loss)
            opt_d.step()
            sch_d.step()


    def configure_optimizers(self):
        optimizer_g = self.hparams.optimizer_g(params=self.generator.parameters())
        optimizer_d = self.hparams.optimizer_d(params=self.discriminator.parameters())
        scheduler_g = self.hparams.scheduler_g(optimizer=optimizer_g)
        scheduler_d = self.hparams.scheduler_d(optimizer=optimizer_d)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]



