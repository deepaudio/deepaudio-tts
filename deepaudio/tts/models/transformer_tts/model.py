from omegaconf import DictConfig
import torch
from torch import Tensor, nn

from pytorch_lightning import LightningModule
from deepaudio.tts.models.transformer_tts import Transformer
from deepaudio.tts.models.transformer_tts.loss import TransformerLoss, GuidedMultiHeadAttentionLoss


class TransformerTTSModel(LightningModule):
    def __init__(self,
                 model: Transformer,
                 modules_applied_guided_attn: str,
                 transformer_loss: TransformerLoss,
                 use_guided_attn_loss: bool,
                 atten_criterion: GuidedMultiHeadAttentionLoss,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler

                 ):
        super(TransformerTTSModel, self).__init__()
        self.model = model

        self.transformer_loss = transformer_loss
        self.modules_applied_guided_attn = modules_applied_guided_attn
        self.use_guided_attn_loss = use_guided_attn_loss
        if self.use_guided_attn_loss:
            self.attn_criterion = atten_criterion

    def training_step(self, batch: dict, batch_idx: int):
        after_outs, before_outs, logits, ys, labels, olens, olens_in, ilens = self.model.forward(
            text=batch['text'],
            text_length=batch['text_length'],
            feats=batch['feats'],
            feats_lengths=batch['feats_lengths'],
            spembs=batch['spembs'],
            sids=batch['spembs'],
            lids=batch['lids']
        )
        # calculate loss values
        l1_loss, l2_loss, bce_loss = self.transformer_loss(
            after_outs, before_outs, logits, ys, labels, olens
        )
        if self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = l2_loss + bce_loss
        elif self.loss_type == "L1+L2":
            loss = l1_loss + l2_loss + bce_loss
        else:
            raise ValueError("unknown --loss-type " + self.loss_type)

        if self.use_guided_attn_loss:
            # calculate for encoder
            if "encoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(
                        reversed(range(len(self.encoder.encoders)))
                ):
                    att_ws += [
                        self.encoder.encoders[layer_idx].self_attn.attn[
                        :, : self.num_heads_applied_guided_attn
                        ]
                    ]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_text, T_text)
                enc_attn_loss = self.attn_criterion(att_ws, ilens, ilens)
                loss = loss + enc_attn_loss
            # calculate for decoder
            if "decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(
                        reversed(range(len(self.decoder.decoders)))
                ):
                    att_ws += [
                        self.decoder.decoders[layer_idx].self_attn.attn[
                        :, : self.num_heads_applied_guided_attn
                        ]
                    ]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_feats, T_feats)
                dec_attn_loss = self.attn_criterion(att_ws, olens_in, olens_in)
                loss = loss + dec_attn_loss
            # calculate for encoder-decoder
            if "encoder-decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(
                        reversed(range(len(self.decoder.decoders)))
                ):
                    att_ws += [
                        self.decoder.decoders[layer_idx].src_attn.attn[
                        :, : self.num_heads_applied_guided_attn
                        ]
                    ]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_feats, T_text)
                enc_dec_attn_loss = self.attn_criterion(att_ws, ilens, olens_in)
                loss = loss + enc_dec_attn_loss
        return {'loss': loss}

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