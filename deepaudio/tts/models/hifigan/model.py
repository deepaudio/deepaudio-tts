import torch
from torch import Tensor, nn

from pytorch_lightning import LightningModule

from deepaudio.tts.models.hifigan.loss import FeatureMatchLoss
from deepaudio.tts.models.hifigan.loss import MelSpectrogramLoss
from deepaudio.tts.models.hifigan.loss import GeneratorAdversarialLoss
from deepaudio.tts.models.hifigan.loss import DiscriminatorAdversarialLoss


class HifiganModel(LightningModule):
    def __init__(self,
                 generator: torch.nn.Module,
                 discriminator: torch.nn.Module,
                 criterion_feat_match: FeatureMatchLoss,
                 criterion_mel: MelSpectrogramLoss,
                 criterion_gen_adv: GeneratorAdversarialLoss,
                 criterion_dis_adv: DiscriminatorAdversarialLoss,
                 lambda_aux: float,
                 lambda_feat_match: float,
                 lambda_adv: float,
                 optimizer_d: torch.optim.Optimizer,
                 scheduler_d: torch.optim.lr_scheduler,
                 optimizer_g: torch.optim.Optimizer,
                 scheduler_g: torch.optim.lr_scheduler,
                 ):
        super(HifiganModel, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.criterion_feat_match = criterion_feat_match
        self.criterion_mel = criterion_mel
        self.criterion_gen_adv = criterion_gen_adv
        self.criterion_dis_adv = criterion_dis_adv
        self.save_hyperparameters(logger=False, ignore=["generator",
                                                        "discriminator",
                                                        "criterion_feat_match",
                                                        "criterion_mel",
                                                        "criterion_gen_adv",
                                                        "criterion_dis_adv"])

    def training_step(self, batch: tuple, batch_idx: int, optimizer_idx: int):
        losses_dict = {}
        # parse batch
        wav, mel = batch

        # Generator
        if optimizer_idx == 0:
            # (B, out_channels, T ** prod(upsample_scales)
            wav_ = self.generator(mel)

            # initialize
            gen_loss = 0.0
            aux_loss = 0.0

            # mel spectrogram loss
            mel_loss = self.criterion_mel(wav_, wav)
            aux_loss += mel_loss
            losses_dict["mel_loss"] = mel_loss

            gen_loss += aux_loss * self.hparams.lambda_aux

            # adversarial loss
            p_ = self.discriminator(wav_)
            adv_loss = self.criterion_gen_adv(p_)
            losses_dict["adversarial_loss"] = adv_loss

            # feature matching loss
            # no need to track gradients
            with torch.no_grad():
                p = self.discriminator(wav)
            fm_loss = self.criterion_feat_match(p_, p)
            losses_dict["feature_matching_loss"] = float(fm_loss)

            adv_loss += self.hparams.lambda_feat_match * fm_loss

            gen_loss += self.hparams.lambda_adv * adv_loss

            losses_dict["generator_loss"] = float(gen_loss)


        # Disctiminator
        if optimizer_idx == 1:
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

    def configure_optimizers(self):
        optimizer_g = self.hparams.optimizer_g(params=self.generator.parameters())
        optimizer_d = self.hparams.optimizer_d(params=self.discriminator.parameters())
        scheduler_g = self.hparams.scheduler_g(optimizer=optimizer_g)
        scheduler_d = self.hparams.scheduler_d(optimizer=optimizer_d)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

