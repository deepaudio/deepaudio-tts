import torch
from torch import Tensor, nn

from pytorch_lightning import LightningModule
from deepaudio.tts.models.tacotron2.tacotron2 import Tacotron2
from deepaudio.tts.models.tacotron2.loss import Tacotron2Loss
from deepaudio.tts.models.tacotron2.loss import GuidedAttentionLoss


class Tacotron2Model(LightningModule):
    def __init__(self,
                 model: Tacotron2,
                 taco2_loss: Tacotron2Loss,
                 use_guided_attn_loss: bool,
                 attn_loss: GuidedAttentionLoss,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler
    ):
        super(Tacotron2Model, self).__init__()

        self.model = model
        self.taco2_loss = taco2_loss
        self.use_guided_attn_loss = use_guided_attn_loss
        if self.use_guided_attn_loss:
            self.attn_loss = attn_loss

    def compute_loss(self, batch):
        losses_dict = {}
        # spk_id!=None in multiple spk fastspeech2
        spk_id = batch["spk_id"] if "spk_id" in batch else None
        spk_emb = batch["spk_emb"] if "spk_emb" in batch else None
        if spk_emb is not None:
            spk_id = None

        after_outs, before_outs, logits, ys, labels, olens, att_ws, olens_in = self.model(
            text=batch["text"],
            text_lengths=batch["text_lengths"],
            speech=batch["speech"],
            speech_lengths=batch["speech_lengths"],
            spk_id=spk_id,
            spk_emb=spk_emb)

        # calculate taco2 loss
        l1_loss, mse_loss, bce_loss = self.taco2_loss(
            after_outs=after_outs,
            before_outs=before_outs,
            logits=logits,
            ys=ys,
            stop_labels=labels,
            olens=olens)

        if self.loss_type == "L1+L2":
            loss = l1_loss + mse_loss + bce_loss
        elif self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = mse_loss + bce_loss
        else:
            raise ValueError(f"unknown --loss-type {self.loss_type}")

        # calculate attention loss
        if self.use_guided_attn_loss:
            # NOTE: length of output for auto-regressive
            # input will be changed when r > 1
            attn_loss = self.attn_loss(
                att_ws=att_ws, ilens=batch["text_lengths"] + 1, olens=olens_in)
            losses_dict["attn_loss"] = attn_loss
            loss = loss + attn_loss

        losses_dict["l1_loss"] = l1_loss
        losses_dict["mse_loss"] = mse_loss
        losses_dict["bce_loss"] = bce_loss
        losses_dict["loss"] = loss
        return losses_dict

    def training_step(self, batch: dict, batch_idx: int):
        losses_dict = self.compute_loss(batch)
        return losses_dict

    def validation_step(self, batch: dict, batch_idx: int):
        losses_dict = self.compute_loss(batch)
        loss = losses_dict.pop('loss')
        losses_dict['val_loss'] = loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return losses_dict

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

