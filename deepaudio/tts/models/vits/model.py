import torch
from torch import Tensor, nn

from pytorch_lightning import LightningModule
from deepaudio.tts.models.vits import VITS
from deepaudio.tts.modules.nets_utils import get_segments
from deepaudio.tts.modules.losses import MelSpectrogramLoss
from deepaudio.tts.modules.losses import FeatureMatchLoss
from deepaudio.tts.modules.losses import GeneratorAdversarialLoss
from deepaudio.tts.modules.losses import DiscriminatorAdversarialLoss

from deepaudio.tts.models.vits.loss import KLDivergenceLoss


class VitsModel(LightningModule):
    def __init__(self,
                 model: VITS,
                 criterion_mel: MelSpectrogramLoss,
                 criterion_feat_match: FeatureMatchLoss,
                 criterion_gen_adv: GeneratorAdversarialLoss,
                 criterion_dis_adv: DiscriminatorAdversarialLoss,
                 lambda_mel: float,
                 lambda_kl: float,
                 lambda_dur: float,
                 lambda_adv: float,
                 lambda_feat_match: float,
                 optimizer_d: torch.optim.Optimizer,
                 scheduler_d: torch.optim.lr_scheduler,
                 optimizer_g: torch.optim.Optimizer,
                 scheduler_g: torch.optim.lr_scheduler,
                 ):
        super(VitsModel, self).__init__()
        self.model = model
        self.criterion_mel = criterion_mel
        self.criterion_feat_match = criterion_feat_match
        self.criterion_gen_adv = criterion_gen_adv
        self.criterion_dis_adv = criterion_dis_adv
        self.criterion_kl = KLDivergenceLoss()
        self.save_hyperparameters(logger=False, ignore=["model", "criterion_mel",
                                                        "criterion_feat_match",
                                                        "criterion_gen_adv",
                                                        "criterion_dis_adv"])

    def step(self, batch, forward_generator):
        return self.model.forward(
            text=batch['text'],
            text_lengths=batch['text_lengths'],
            feats=batch['feats'],
            feats_lengths=batch['feats_lengths'],
            speech=batch['speech'],
            speech_lengths=batch['speech_lengths'],
            sids=batch['sids'],
            spembs=batch['spembs'],
            lids=batch['lids'],
            forward_generator=forward_generator,
        )

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int):
        # Generator
        if optimizer_idx == 0:
            forward_generator = True
            outs = self.step(batch, forward_generator)
            # parse outputs
            speech_hat_, dur_nll, _, start_idxs, _, z_mask, outs_ = outs
            _, z_p, m_p, logs_p, _, logs_q = outs_
            speech_ = get_segments(
                x=batch['speech'],
                start_idxs=start_idxs * self.generator.upsample_factor,
                segment_size=self.generator.segment_size * self.generator.upsample_factor,
            )

            # calculate discriminator outputs
            p_hat = self.discriminator(speech_hat_)
            with torch.no_grad():
                # do not store discriminator gradient in generator turn
                p = self.model.discriminator(speech_)

            # calculate losses
            mel_loss = self.mel_loss(speech_hat_, speech_)
            kl_loss = self.kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
            dur_loss = torch.sum(dur_nll.float())
            adv_loss = self.generator_adv_loss(p_hat)
            feat_match_loss = self.feat_match_loss(p_hat, p)

            mel_loss = mel_loss * self.hparams.lambda_mel
            kl_loss = kl_loss * self.hparams.lambda_kl
            dur_loss = dur_loss * self.hparams.lambda_dur
            adv_loss = adv_loss * self.hparams.lambda_adv
            feat_match_loss = feat_match_loss * self.hparams.lambda_feat_match
            loss = mel_loss + kl_loss + dur_loss + adv_loss + feat_match_loss
            return {'loss': loss}

        # Disctiminator
        if optimizer_idx == 1:
            forward_generator = False
            forward_generator = True
            outs = self.step(batch, forward_generator)
            # parse outputs
            speech_hat_, _, _, start_idxs, *_ = outs
            speech_ = get_segments(
                x=batch['speech'],
                start_idxs=start_idxs * self.model.generator.upsample_factor,
                segment_size=self.model.generator.segment_size * self.model.generator.upsample_factor,
            )

            # calculate discriminator outputs
            # TODO: check no grad
            p_hat = self.discriminator(speech_hat_.detach())
            p = self.discriminator(speech_)

            # calculate losses
            real_loss, fake_loss = self.discriminator_adv_loss(p_hat, p)
            loss = real_loss + fake_loss
            return {'loss': loss}

    def configure_optimizers(self):
        optimizer_g = self.hparams.optimizer_g(params=self.generator.parameters())
        optimizer_d = self.hparams.optimizer_d(params=self.discriminator.parameters())
        scheduler_g = self.hparams.scheduler_g(optimizer=optimizer_g)
        scheduler_d = self.hparams.scheduler_d(optimizer=optimizer_d)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]


