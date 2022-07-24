from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from deepaudio.tts.dataclass.configurations import DeepMMDataclass


@dataclass
class TransformerConfigs(DeepMMDataclass):
    name: str = field(
        default="transformer", metadata={"help": "Model name"}
    )
    idim: int = field(
        default=1, metadata={"help": "Dimension of the inputs."}
    )
    odim: int = field(
        default=1, metadata={"help": "Dimension of the outputs."}
    )
    embed_dim: int = field(
        default=512, metadata={"help": "Dimension of character embedding."}
    )
    eprenet_conv_layers: int = field(
        default=3, metadata={"help": "Number of encoder prenet convolution layers."}
    )
    eprenet_conv_chans: int = field(
        default=256, metadata={"help": "Number of encoder prenet convolution channels."}
    )
    eprenet_conv_filts: int = field(
        default=5, metadata={"help": "Filter size of encoder prenet convolution."}
    )
    dprenet_layers: int = field(
        default=2, metadata={"help": "Number of decoder prenet layers."}
    )
    dprenet_units: int = field(
        default=256, metadata={"help": "Number of decoder prenet hidden units."}
    )
    elayers: int = field(
        default=6, metadata={"help": "Number of encoder layers."}
    )
    eunits: int = field(
        default=1024, metadata={"help": "Number of encoder hidden units."}
    )
    adim: int = field(
        default=512, metadata={"help": "Number of attention transformation dimensions."}
    )
    aheads: int = field(
        default=4, metadata={"help": "Number of heads for multi head attention."}
    )
    dlayers: int = field(
        default=6, metadata={"help": "Number of decoder layers."}
    )
    dunits: int = field(
        default=1024, metadata={"help": "Number of decoder hidden units."}
    )
    postnet_layers: int = field(
        default=5, metadata={"help": "Number of postnet layers."}
    )
    postnet_chans: int = field(
        default=256, metadata={"help": "Number of postnet channels."}
    )
    postnet_filts: int = field(
        default=5, metadata={"help": "Filter size of postnet."}
    )
    positionwise_layer_type: str = field(
        default="conv1d", metadata={"help": "Position-wise operation type."}
    )
    positionwise_conv_kernel_size: int = field(
        default=1, metadata={"help": "Kernel size in position wise conv 1d."}
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
    spks: Optional[int] = field(
        default=None, metadata={
            "help": "Number of speakers. If set to > 1, assume that the sids will be provided as the input and use sid embedding layer."}
    )
    langs: Optional[int] = field(
        default=None, metadata={
            "help": "Number of languages. If set to > 1, assume that the lids will be provided as the input and use sid embedding layer."}
    )
    spk_embed_dim: Optional[int] = field(
        default=None, metadata={
            "help": "Speaker embedding dimension. If set to > 0, assume that spembs will be provided as the input."}
    )
    spk_embed_integration_type: str = field(
        default="add", metadata={"help": "How to integrate speaker embedding."}
    )
    use_gst: str = field(
        default=False, metadata={"help": "Whether to use global style token."}
    )
    gst_tokens: int = field(
        default=10, metadata={"help": "Number of GST embeddings."}
    )
    gst_heads: int = field(
        default=4, metadata={"help": "Number of heads in GST multihead attention."}
    )
    gst_conv_layers: int = field(
        default=6, metadata={"help": "Number of conv layers in GST."}
    )
    gst_conv_chans_list: List[int] = field(
        default=(32, 32, 64, 64, 128, 128), metadata={"help": "List of the number of channels of conv layers in GST."}
    )
    gst_conv_kernel_size: int = field(
        default=3, metadata={"help": "Kernel size of conv layers in GST."}
    )
    gst_conv_stride: int = field(
        default=2, metadata={"help": "Stride size of conv layers in GST."}
    )
    gst_gru_layers: int = field(
        default=1, metadata={"help": "Number of GRU layers in GST."}
    )
    gst_gru_units: int = field(
        default=128, metadata={"help": "Number of GRU units in GST."}
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
        default=0.1, metadata={"help": "Dropout rate in decoder except attention & positional encoding."}
    )
    transformer_dec_positional_dropout_rate: float = field(
        default=0.1, metadata={"help": "Dropout rate after decoder positional encoding."}
    )
    transformer_dec_attn_dropout_rate: float = field(
        default=0.1, metadata={"help": "Dropout rate in decoder self-attention module."}
    )
    transformer_enc_dec_attn_dropout_rate: float = field(
        default=0.1, metadata={"help": "Dropout rate in source attention module."}
    )
    eprenet_dropout_rate: float = field(
        default=0.5, metadata={"help": "Dropout rate in encoder prenet."}
    )
    dprenet_dropout_rate: float = field(
        default=0.5, metadata={"help": "Dropout rate in decoder prenet."}
    )
    postnet_dropout_rate: float = field(
        default=0.5, metadata={"help": "Dropout rate in postnet."}
    )
    init_type: str = field(
        default="xavier_uniform", metadata={"help": "How to initialize transformer parameters."}
    )
    init_enc_alpha: float = field(
        default=1.0, metadata={"help": "Initial value of alpha in scaled pos encoding of the encoder."}
    )
    init_dec_alpha: float = field(
        default=1.0, metadata={"help": "Initial value of alpha in scaled pos encoding of the decoder."}
    )
    use_masking: bool = field(
        default=False, metadata={"help": "Whether to apply masking for padded part in loss calculation."}
    )
    use_weighted_masking: bool = field(
        default=False, metadata={"help": "Whether to apply weighted masking in loss calculation."}
    )
    bce_pos_weight: float = field(
        default=5.0, metadata={"help": "Positive sample weight in bce calculation, only for use_masking=true."}
    )
    loss_type: str = field(
        default="L1", metadata={"help": "How to calculate loss."}
    )
    use_guided_attn_loss: bool = field(
        default=True, metadata={"help": "Whether to use guided attention loss."}
    )
    num_heads_applied_guided_attn: int = field(
        default=2, metadata={"help": "Number of heads in each layer to apply guided attention loss."}
    )
    num_layers_applied_guided_attn: int = field(
        default=2, metadata={"help": "Number of layers to apply guided attention loss."}
    )
    modules_applied_guided_attn: List[str] = field(
        default=("encoder-decoder"), metadata={"help": "List of module names to apply guided attention loss."}
    )
    guided_attn_loss_sigma: float = field(
        default=0.4, metadata={"help": "Sigma in guided attention loss."}
    )
    guided_attn_loss_lambda: float = field(
        default=1.0, metadata={"help": "Lambda in guided attention loss."}
    )
    optimizer: str = field(
        default="adam", metadata={"help": "Optimizer for training."}
    )
