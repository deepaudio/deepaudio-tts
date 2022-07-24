from omegaconf import DictConfig
import torch
from torch import Tensor, nn

from deepaudio.tts.models import register_model
from deepaudio.tts.models.base import BasePLModel
from deepaudio.tts.models.transformer_tts.transformer import Transformer
from deepaudio.tts.models.transformer_tts.loss import TransformerLoss

from .configurations import TransformerConfigs


@register_model('transformer_tts', dataclass=TransformerConfigs)
class TransformerTTSModel(BasePLModel):
    def __init__(self, configs: DictConfig):
        super(TransformerTTSModel, self).__init__(configs)

    def build_model(self):
        self.model = Transformer(
            idim=self.configs.model.idim,
            odim=self.configs.model.odim,
            embed_dim=self.configs.model.embed_dim,
            eprenet_conv_layers=self.configs.model.eprenet_conv_layers,
            eprenet_conv_chans=self.configs.model.eprenet_conv_chans,
            eprenet_conv_filts=self.configs.model.eprenet_conv_filts,
            dprenet_layers=self.configs.model.dprenet_layers,
            dprenet_units=self.configs.model.dprenet_units,
            elayers=self.configs.model.elayers,
            eunits=self.configs.model.eunits,
            adim=self.configs.model.adim,
            aheads=self.configs.model.aheads,
            dlayers=self.configs.model.dlayers,
            dunits=self.configs.model.dunits,
            postnet_layers=self.configs.model.postnet_layers,
            postnet_chans=self.configs.model.postnet_chans,
            postnet_filts=self.configs.model.postnet_filts,
            positionwise_layer_type=self.configs.model.positionwise_layer_type,
            positionwise_conv_kernel_size=self.configs.model.positionwise_conv_kernel_size,
            use_scaled_pos_enc=self.configs.model.use_scaled_pos_enc,
            use_batch_norm=self.configs.model.use_batch_norm,
            encoder_normalize_before=self.configs.model.encoder_normalize_before,
            decoder_normalize_before=self.configs.model.decoder_normalize_before,
            encoder_concat_after=self.configs.model.encoder_concat_after,
            decoder_concat_after=self.configs.model.decoder_concat_after,
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
            transformer_enc_dropout_rate=self.configs.model.transformer_enc_dropout_rate,
            transformer_enc_positional_dropout_rate=self.configs.model.transformer_enc_positional_dropout_rate,
            transformer_enc_attn_dropout_rate=self.configs.model.transformer_enc_attn_dropout_rate,
            transformer_dec_dropout_rate=self.configs.model.transformer_dec_dropout_rate,
            transformer_dec_positional_dropout_rate=self.configs.model.transformer_dec_positional_dropout_rate,
            transformer_dec_attn_dropout_rate=self.configs.model.transformer_dec_attn_dropout_rate,
            transformer_enc_dec_attn_dropout_rate=self.configs.model.transformer_enc_dec_attn_dropout_rate,
            eprenet_dropout_rate=self.configs.model.eprenet_dropout_rate,
            dprenet_dropout_rate=self.configs.model.dprenet_dropout_rate,
            postnet_dropout_rate=self.configs.model.postnet_dropout_rate,
            init_type=self.configs.model.init_type,
            init_enc_alpha=self.configs.model.init_enc_alpha,
            init_dec_alpha=self.configs.model.init_dec_alpha,
            use_masking=self.configs.model.use_masking,
            use_weighted_masking=self.configs.model.use_weighted_masking,
            bce_pos_weight=self.configs.model.bce_pos_weight,
            loss_type=self.configs.model.loss_type,
            use_guided_attn_loss=self.configs.model.use_guided_attn_loss,
            num_heads_applied_guided_attn=self.configs.model.num_heads_applied_guided_attn,
            num_layers_applied_guided_attn=self.configs.model.num_layers_applied_guided_attn,
            modules_applied_guided_attn=self.configs.model.modules_applied_guided_attn,
            guided_attn_loss_sigma=self.configs.model.guided_attn_loss_sigma,
            guided_attn_loss_lambda=self.configs.model.guided_attn_loss_lambda,

        )

    def configure_criterion(self):
        self.criterion = TransformerLoss(
            use_masking=self.config.model.use_masking,
            use_weighted_masking=self.config.model.use_weighted_masking,
            bce_pos_weight=self.config.model.bce_pos_weight,
        )
        if self.config.model.use_guided_attn_loss:
            self.attn_criterion = self.config.model.GuidedMultiHeadAttentionLoss(
                sigma=self.config.model.guided_attn_loss_sigma,
                alpha=self.config.model.guided_attn_loss_lambda,
            )

    def training_step(self, batch: dict, batch_idx: int):
        after_outs, before_outs, logits, ys, labels, olens = self.model.forward(
            text=batch['text'],
            text_length=batch['text_length'],
            feats=batch['feats'],
            feats_lengths=batch['feats_lengths'],
            spembs=batch['spembs'],
            sids=batch['spembs'],
            lids=batch['lids']
        )
        # calculate loss values
        l1_loss, l2_loss, bce_loss = self.criterion(
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

        stats = dict(
            l1_loss=l1_loss.item(),
            l2_loss=l2_loss.item(),
            bce_loss=bce_loss.item(),
        )
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
                stats.update(enc_attn_loss=enc_attn_loss.item())
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
                stats.update(dec_attn_loss=dec_attn_loss.item())
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
                stats.update(enc_dec_attn_loss=enc_dec_attn_loss.item())

        # report extra information
        if self.use_scaled_pos_enc:
            stats.update(
                encoder_alpha=self.encoder.embed[-1].alpha.data.item(),
                decoder_alpha=self.decoder.embed[-1].alpha.data.item(),
            )




