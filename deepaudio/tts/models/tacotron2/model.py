from omegaconf import DictConfig
from torch import Tensor, nn

from deepaudio.tts.models import register_model
from deepaudio.tts.models.base import BasePLModel
from deepaudio.tts.models.tacotron2 import Tacotron2
from deepaudio.tts.models.tacotron2.loss import Tacotron2Loss
from deepaudio.tts.models.tacotron2.loss import GuidedAttentionLoss


from .configurations import Tacotron2Configs


@register_model('tacotron2', dataclass=Tacotron2Configs)
class Tacotron2Model(BasePLModel):
    def __init__(self, configs: DictConfig):
        super(Tacotron2Model, self).__init__(configs)

    def build_model(self):
        self.model = Tacotron2(
            idim=self.configs.model.idim,
            odim=self.configs.model.odim,
            embed_dim=self.configs.model.embed_dim,
            elayers=self.configs.model.elayers,
            eunits=self.configs.model.eunits,
            econv_layers=self.configs.model.econv_layers,
            econv_chans=self.configs.model.econv_chans,
            econv_filts=self.configs.model.econv_filts,
            atype=self.configs.model.atype,
            adim=self.configs.model.adim,
            aconv_chans=self.configs.model.aconv_chans,
            aconv_filts=self.configs.model.aconv_filts,
            cumulate_att_w=self.configs.model.cumulate_att_w,
            dlayers=self.configs.model.dlayers,
            dunits=self.configs.model.dunits,
            prenet_layers=self.configs.model.prenet_layers,
            prenet_units=self.configs.model.prenet_units,
            postnet_layers=self.configs.model.postnet_layers,
            postnet_chans=self.configs.model.postnet_chans,
            postnet_filts=self.configs.model.postnet_filts,
            output_activation=self.configs.model.output_activation,
            use_batch_norm=self.configs.model.use_batch_norm,
            use_concate=self.configs.model.use_concate,
            use_residual=self.configs.model.use_residual,
            reduction_factor=self.configs.model.reduction_factor,
            spks=self.configs.model.spks,
            langs=self.configs.model.langs,
            spk_embed_dim=self.configs.model.spk_embed_dim,
            spk_embed_integration_type=self.configs.model.spk_embed_integration_type,
            use_gst=self.configs.model.use_gst,
            gst_tokens=self.configs.model.gst_tokens,
            gst_heads=self.configs.model.gst_heads,
            gst_conv_layers=self.configs.model.gst_conv_layers,
            gst_conv_chans_list=self.configs.model.gst_conv_chans_list,
            gst_conv_kernel_size=self.configs.model.gst_conv_kernel_size,
            gst_conv_stride=self.configs.model.gst_conv_stride,
            gst_gru_layers=self.configs.model.gst_gru_layers,
            gst_gru_units=self.configs.model.gst_gru_units,
            dropout_rate=self.configs.model.dropout_rate,
            zoneout_rate=self.configs.model.zoneout_rate,
            use_masking=self.configs.model.use_masking,
            use_weighted_masking=self.configs.model.use_weighted_masking,
            bce_pos_weight=self.configs.model.bce_pos_weight,
            loss_type=self.configs.model.loss_type,
            use_guided_attn_loss=self.configs.model.use_guided_attn_loss,
            guided_attn_loss_sigma=self.configs.model.guided_attn_loss_sigma,
            guided_attn_loss_lambda=self.configs.model.guided_attn_loss_lambda
        )

    def configure_criterion(self):
        self.taco2_loss = Tacotron2Loss(
            use_masking=self.configs.model.use_masking,
            use_weighted_masking=self.configs.model.use_weighted_masking,
            bce_pos_weight=self.configs.model.bce_pos_weight,
        )
        if self.use_guided_attn_loss:
            self.attn_loss = GuidedAttentionLoss(
                sigma=self.configs.model.guided_attn_loss_sigma,
                alpha=self.configs.model.guided_attn_loss_lambda,
            )

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
        return losses_dict


