from dataclasses import dataclass, field
from typing import Optional, Sequence

from deepaudio.tts.dataclass.configurations import DeepMMDataclass


@dataclass
class Fastspeech2Configs(DeepMMDataclass):
    name: str = field(
        default="fastspeech2", metadata={"help": "Model name"}
    )
    idim: int = field(
        default=1, metadata={"help": "Dimension of the inputs."}
    )
    odim: int = field(
        default=1, metadata={"help": "Dimension of the outputs."}
    )
    adim: int = field(
        default=384, metadata={"help": "Dimension of the attention."}
    )
    aheads: int = field(
        default=4, metadata={"help": "Number of attention heads."}
    )
    elayers: int = field(
        default=6, metadata={"help": "Number of encoder layers."}
    )
    eunits: int = field(
        default=1536, metadata={"help": "Number of encoder hidden units."}
    )
    dlayers: int = field(
        default=6, metadata={"help": "Number of decoder layers."}
    )
    dunits: int = field(
        default=1536, metadata={"help": "Number of decoder hidden units."}
    )
    postnet_layers: int = field(
        default=5, metadata={"help": "Number of postnet layers."}
    )
    postnet_chans: int = field(
        default=512, metadata={"help": "Number of postnet channels."}
    )
    postnet_filts: int = field(
        default=5, metadata={"help": "Kernel size of postnet."}
    )
    postnet_dropout_rate: float = field(
        default=0.5, metadata={"help": "Dropout rate in postnet."}
    )

    positionwise_layer_type: str = field(
        default="conv1d", metadata={"help": "positionwise_layer_type"}
    )
    positionwise_conv_kernel_size: int = field(
        default=1, metadata={"help": "positionwise conv kernel size"}
    )
    use_scaled_pos_enc: bool = field(
        default=True, metadata={"help": "Whether to use trainable scaled pos encoding."}
    )
    use_batch_norm: bool = field(
        default=True, metadata={"help": "Whether to use batch normalization in encoder prenet."}
    )

    encoder_normalize_before: bool = field(
        default=True, metadata={"help": "Whether to apply layernorm layer before encoder block."}
    )
    decoder_normalize_before: bool = field(
        default=True, metadata={"help": "Whether to apply layernorm layer before decoder block."}
    )
    encoder_concat_after: bool = field(
        default=False, metadata={"help": "Whether to concatenate attention layer's input and output in encoder."}
    )
    decoder_concat_after: bool = field(
        default=False, metadata={"help": "Whether to concatenate attention layer's input and output in decoder."}
    )

    reduction_factor: int = field(
        default=1, metadata={"help": "Reduction factor."}
    )
    encoder_type: str = field(
        default="transformer", metadata={"help": "Encoder type (transformer or conformer)."}
    )
    decoder_type: str = field(
        default="transformer", metadata={"help": "Decoder type (transformer or conformer)."}
    )
    transformer_enc_dropout_rate: float = field(
        default=0.1, metadata={"help": "Dropout rate in encoder except attention and positional encoding."}
    )
    transformer_enc_positional_dropout_rate: float = field(
        default=0.1, metadata={"help": "Dropout rate after encoder positional encoding."}
    )
    transformer_enc_attn_dropout_rate: float = field(
        default=0.1, metadata={"help": "Dropout rate in encoder self-attention module."}
    )
    transformer_dec_dropout_rate: float = field(
        default=0.1, metadata={"help": "Dropout rate in decoder except attention and positional encoding."}
    )
    transformer_dec_positional_dropout_rate: float = field(
        default=0.1, metadata={"help": "Dropout rate after decoder positional encoding."}
    )
    transformer_dec_attn_dropout_rate: float = field(
        default=0.1, metadata={"help": "Dropout rate in decoder self-attention module."}
    )
    conformer_rel_pos_type: str = field(
        default="legacy", metadata={"help": "Relative pos encoding type in conformer."}
    )
    conformer_pos_enc_layer_type: str = field(
        default="rel_pos", metadata={"help": " Pos encoding layer type in conformer."}
    )
    conformer_self_attn_layer_type: str = field(
        default="rel_selfattn", metadata={"help": " Self-attention layer type in conformer"}
    )
    conformer_activation_type: str = field(
        default="swish", metadata={"help": " Activation function type in conformer."}
    )
    use_macaron_style_in_conformer: bool = field(
        default=True, metadata={"help": "Whether to use macaron style FFN."}
    )
    use_cnn_in_conformer: bool = field(
        default=True, metadata={"help": "Whether to use CNN in conformer."}
    )
    zero_triu: bool = field(
        default=False, metadata={"help": "Whether to use zero triu in relative self-attention module."}
    )
    conformer_enc_kernel_size: int = field(
        default=7, metadata={"help": "Kernel size of encoder conformer."}
    )
    conformer_dec_kernel_size: int = field(
        default=31, metadata={"help": "Kernel size of decoder conformer."}
    )

    duration_predictor_layers: int = field(
        default=2, metadata={"help": " Number of duration predictor layers."}
    )
    duration_predictor_chans: int = field(
        default=384, metadata={"help": " Number of duration predictor channels."}
    )
    duration_predictor_kernel_size: int = field(
        default=3, metadata={"help": " Kernel size of duration predictor."}
    )
    duration_predictor_dropout_rate: float = field(
        default=0.1, metadata={"help": " Dropout rate in duration predictor."}
    )
    pitch_predictor_layers: int = field(
        default=2, metadata={"help": " Number of pitch predictor layers."}
    )
    pitch_predictor_chans: int = field(
        default=384, metadata={"help": " Number of pitch predictor channels."}
    )
    pitch_predictor_kernel_size: int = field(
        default=3, metadata={"help": " Kernel size of pitch predictor."}
    )
    pitch_predictor_dropout: float = field(
        default=0.5, metadata={"help": " Dropout rate in pitch predictor."}
    )
    pitch_embed_kernel_size: float = field(
        default=9, metadata={"help": " Kernel size of pitch embedding."}
    )
    pitch_embed_dropout: float = field(
        default=0.5, metadata={"help": " Dropout rate for pitch embedding."}
    )
    stop_gradient_from_pitch_predictor: bool = field(
        default=False, metadata={"help": "Whether to stop gradient from pitch predictor to encoder."}
    )
    energy_predictor_layers: int = field(
        default=2, metadata={"help": " Number of energy predictor layers."}
    )
    energy_predictor_chans: int = field(
        default=384, metadata={"help": " Number of energy predictor channels."}
    )
    energy_predictor_kernel_size: int = field(
        default=3, metadata={"help": " Kernel size of energy predictor."}
    )
    energy_predictor_dropout: float = field(
        default=0.5, metadata={"help": " Dropout rate in energy predictor."}
    )
    energy_embed_kernel_size: float = field(
        default=9, metadata={"help": " Kernel size of energy embedding."}
    )
    energy_embed_dropout_rate: float = field(
        default=0.5, metadata={"help": " Dropout rate for energy embedding."}
    )
    stop_gradient_from_energy_predictor: bool = field(
        default=False, metadata={"help": " Whether to stop gradient from pitch predictor to encoder."}
    )

    spks: Optional[int] = field(
        default=None, metadata={"help": " Number of speakers."}
    )
    langs: Optional[int] = field(
        default=None, metadata={"help": "Number of languages. If set to > 1, assume that the \
                lids will be provided as the input and use sid embedding layer.."}
    )
    spk_embed_dim: Optional[int] = field(
        default=None, metadata={"help": " Speaker embedding dimension."}
    )
    spk_embed_integration_type: str = field(
        default="add", metadata={"help": "How to integrate speaker embedding.."}
    )
    use_gst: bool = field(
        default=False, metadata={"help": " Whether to use global style token."}
    )
    gst_tokens: int = field(
        default=10, metadata={"help": " The number of GST embeddings."}
    )
    gst_heads: int = field(
        default=4, metadata={"help": " The number of heads in GST multihead attention."}
    )
    gst_conv_layers: int = field(
        default=6, metadata={"help": " The number of conv layers in GST."}
    )
    gst_conv_chans_list: Sequence[int] = field(
        default=(32, 32, 64, 64, 128, 128), metadata={"help": "List of the number of channels of conv layers in GST."}
    )
    gst_conv_kernel_size: int = field(
        default=3, metadata={"help": " Kernel size of conv layers in GST."}
    )
    gst_conv_stride: int = field(
        default=2, metadata={"help": " Stride size of conv layers in GST."}
    )
    gst_gru_layers: int = field(
        default=1, metadata={"help": " The number of GRU layers in GST."}
    )
    gst_gru_units: int = field(
        default=128, metadata={"help": " The number of GRU units in GST."}
    )
    init_type: str = field(
        default="xavier_uniform", metadata={"help": " How to initialize transformer parameters."}
    )
    init_enc_alpha: float = field(
        default=1.0, metadata={"help": " Initial value of alpha in scaled pos encoding of the encoder."}
    )
    init_dec_alpha: float = field(
        default=1.0, metadata={"help": " Initial value of alpha in scaled pos encoding of the decoder."}
    )
    use_masking: bool = field(
        default=False, metadata={"help": " Whether to apply masking for padded part in loss calculation."}
    )
    use_weighted_masking: bool = field(
        default=False, metadata={"help": " Whether to apply weighted masking in loss calculation."}
    )
    optimizer: str = field(
        default="adam", metadata={"help": "Optimizer for training."}
    )