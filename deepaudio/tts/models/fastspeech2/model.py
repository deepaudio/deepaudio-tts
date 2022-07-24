from omegaconf import DictConfig
from torch import Tensor, nn

from deepaudio.tts.models import register_model
from deepaudio.tts.models.base import BasePLModel
from deepaudio.tts.models.fastspeech2.fastspeech2 import FastSpeech2
from deepaudio.tts.models.fastspeech2.loss import FastSpeech2Loss

from .configurations import Fastspeech2Configs


@register_model('fastspeech2', dataclass=Fastspeech2Configs)
class Fastspeech2Model(BasePLModel):
    def __init__(self, configs: DictConfig, num_classes: int):
        super(Fastspeech2Model, self).__init__(configs, num_classes)

    def build_model(self):
        self.model = FastSpeech2(
            idim=self.configs.model.idim,
            odim=self.configs.model.odim,
            adim=self.configs.model.adim,
            aheads=self.configs.model.aheads,
            elayers=self.configs.model.elayers,
            eunits=self.configs.model.eunits,
            dlayers=self.configs.model.dlayers,
            dunits=self.configs.model.dunits,
            postnet_layers=self.configs.model.postnet_layers,
            postnet_chans=self.configs.model.postnet_chans,
            postnet_filts=self.configs.model.postnet_filts,
            postnet_dropout_rate=self.configs.model.postnet_dropout_rate,
            positionwise_layer_type=self.configs.model.positionwise_layer_type,
            positionwise_conv_kernel_size=self.configs.model.positionwise_conv_kernel_size,
            use_scaled_pos_enc=self.configs.model.use_scaled_pos_enc,
            use_batch_norm=self.configs.model.use_batch_norm,
            encoder_normalize_before=self.configs.model.encoder_normalize_before,
            decoder_normalize_before=self.configs.model.decoder_normalize_before,
            encoder_concat_after=self.configs.model.encoder_concat_after,
            decoder_concat_after=self.configs.model.decoder_concat_after,
            reduction_factor=self.configs.model.reduction_factor,
            encoder_type=self.configs.model.encoder_type,
            decoder_type=self.configs.model.decoder_type,
            transformer_enc_dropout_rate=self.configs.model.transformer_enc_dropout_rate,
            transformer_enc_positional_dropout_rate=self.configs.model.transformer_enc_positional_dropout_rate,
            transformer_enc_attn_dropout_rate=self.configs.model.transformer_enc_attn_dropout_rate,
            transformer_dec_dropout_rate=self.configs.model.transformer_dec_dropout_rate,
            transformer_dec_positional_dropout_rate=self.configs.model.transformer_dec_positional_dropout_rate,
            transformer_dec_attn_dropout_rate=self.configs.model.transformer_dec_attn_dropout_rate,
            conformer_rel_pos_type=self.configs.model.conformer_rel_pos_type,
            conformer_pos_enc_layer_type=self.configs.model.conformer_pos_enc_layer_type,
            conformer_self_attn_layer_type=self.configs.model.conformer_self_attn_layer_type,
            conformer_activation_type=self.configs.model.conformer_activation_type,
            use_macaron_style_in_conformer=self.configs.model.use_macaron_style_in_conformer,
            use_cnn_in_conformer=self.configs.model.use_cnn_in_conformer,
            zero_triu=self.configs.model.zero_triu,
            conformer_enc_kernel_size=self.configs.model.conformer_enc_kernel_size,
            conformer_dec_kernel_size=self.configs.model.conformer_dec_kernel_size,
            duration_predictor_layers=self.configs.model.duration_predictor_layers,
            duration_predictor_chans=self.configs.model.duration_predictor_chans,
            duration_predictor_kernel_size=self.configs.model.duration_predictor_kernel_size,
            duration_predictor_dropout_rate=self.configs.model.duration_predictor_dropout_rate,
            energy_predictor_layers=self.configs.model.energy_predictor_layers,
            energy_predictor_chans=self.configs.model.energy_predictor_chans,
            energy_predictor_kernel_size=self.configs.model.energy_predictor_kernel_size,
            energy_predictor_dropout=self.configs.model.energy_predictor_dropout,
            energy_embed_kernel_size=self.configs.model.energy_embed_kernel_size,
            energy_embed_dropout=self.configs.model.energy_embed_dropout,
            stop_gradient_from_energy_predictor=self.configs.model.stop_gradient_from_energy_predictor,
            pitch_predictor_layers=self.configs.model.pitch_predictor_layers,
            pitch_predictor_chans=self.configs.model.pitch_predictor_chans,
            pitch_predictor_kernel_size=self.configs.model.pitch_predictor_kernel_size,
            pitch_predictor_dropout=self.configs.model.pitch_predictor_dropout,
            pitch_embed_kernel_size=self.configs.model.pitch_embed_kernel_size,
            pitch_embed_dropout=self.configs.model.pitch_embed_dropout,
            stop_gradient_from_pitch_predictor=self.configs.model.stop_gradient_from_pitch_predictor,
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
            init_type=self.configs.model.init_type,
            init_enc_alpha=self.configs.model.init_enc_alpha,
            init_dec_alpha=self.configs.model.init_dec_alpha,
            use_masking=self.configs.model.use_masking,
            use_weighted_masking=self.configs.model.use_weighted_masking,
        )

    def configure_criterion(self) -> nn.Module:
        self.criterion = FastSpeech2Loss()

    def training_step(self, batch: tuple, batch_idx: int):
        losses_dict = {}
        # spk_id!=None in multiple spk fastspeech2
        spk_id = batch["spk_id"] if "spk_id" in batch else None
        spk_emb = batch["spk_emb"] if "spk_emb" in batch else None
        # No explicit speaker identifier labels are used during voice cloning training.
        if spk_emb is not None:
            spk_id = None

        before_outs, after_outs, d_outs, p_outs, e_outs, ys, olens = self.model(
            text=batch["text"],
            text_lengths=batch["text_lengths"],
            speech=batch["speech"],
            speech_lengths=batch["speech_lengths"],
            durations=batch["durations"],
            pitch=batch["pitch"],
            energy=batch["energy"],
            spk_id=spk_id,
            spk_emb=spk_emb)

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
