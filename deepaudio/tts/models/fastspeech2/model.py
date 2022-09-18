import torch
from torch import Tensor, nn
from pytorch_lightning import LightningModule

from deepaudio.tts.models.fastspeech2 import FastSpeech2
from deepaudio.tts.models.fastspeech2.loss import FastSpeech2Loss


class Fastspeech2Model(LightningModule):
    def __init__(self,
                 model: FastSpeech2,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler, ):
        super(Fastspeech2Model, self).__init__()

        self.save_hyperparameters(logger=False, ignore=["model"])
        self.model = model
        self.criterion = FastSpeech2Loss()

    def step(self, batch):
        # spk_id!=None in multiple spk fastspeech2
        spk_id = batch["spk_id"] if "spk_id" in batch else None
        spk_emb = batch["spk_emb"] if "spk_emb" in batch else None
        # No explicit speaker identifier labels are used during voice cloning training.
        if spk_emb is not None:
            spk_id = None

        outs = self.model(
            text=batch["text"],
            text_lengths=batch["text_lengths"],
            speech=batch["speech"],
            speech_lengths=batch["speech_lengths"],
            durations=batch["durations"],
            pitch=batch["pitch"],
            energy=batch["energy"],
            spk_id=spk_id,
            spk_emb=spk_emb)
        return outs

    def training_step(self, batch: dict, batch_idx: int):
        before_outs, after_outs, d_outs, p_outs, e_outs, ys, olens = self.step(batch)
        l1_loss, duration_loss, pitch_loss, energy_loss = self.criterion(
            after_outs=after_outs,
            before_outs=before_outs,
            d_outs=d_outs,
            p_outs=p_outs,
            e_outs=e_outs,
            ys=ys,
            ds=batch["durations"],
            ps=batch["pitch"],
            es=batch["energy"],
            ilens=batch["text_lengths"],
            olens=olens)

        loss = l1_loss + duration_loss + pitch_loss + energy_loss
        return {'loss': loss}

    def validation_step(self, batch: dict, batch_idx: int):
        before_outs, after_outs, d_outs, p_outs, e_outs, ys, olens = self.step(batch)
        l1_loss, duration_loss, pitch_loss, energy_loss = self.criterion(
            after_outs=after_outs,
            before_outs=before_outs,
            d_outs=d_outs,
            p_outs=p_outs,
            e_outs=e_outs,
            ys=ys,
            ds=batch["durations"],
            ps=batch["pitch"],
            es=batch["energy"],
            ilens=batch["text_lengths"],
            olens=olens)

        loss = l1_loss + duration_loss + pitch_loss + energy_loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
