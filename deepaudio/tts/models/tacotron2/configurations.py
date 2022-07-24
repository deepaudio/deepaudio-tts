from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from deepaudio.tts.dataclass.configurations import DeepMMDataclass


@dataclass
class Tacotron2Configs(DeepMMDataclass):
    name: str = field(
        default="tacotron2", metadata={"help": "Model name"}
    )
    idim: int = field(
        default=1, metadata={"help": "Dimension of the inputs"}
    )
    odim: int = field(
        default=1, metadata={"help": "Dimension of the outputs"}
    )
    embed_dim: int = field(
        default=512, metadata={"help": "Dimension of the token embedding"}
    )
    elayers: int = field(
        default=1, metadata={"help": "Number of encoder blstm layers"}
    )
    eunits: int = field(
        default=512, metadata={"help": "Number of encoder blstm units"}
    )
    econv_layers: int = field(
        default=3, metadata={"help": "Number of encoder conv layers"}
    )
    econv_filts: int = field(
        default=5, metadata={"help": "Number of encoder conv filter size"}
    )
    econv_chans: int = field(
        default=512, metadata={"help": "Number of encoder conv filter channels"}
    )
    atype: str = field(
        default="location", metadata={"help": "Attention type."}
    )
    adim: int = field(
        default=512, metadata={"help": "Number of dimension of mlp in attention"}
    )
    aconv_chans: int = field(
        default=32, metadata={"help": "Number of attention conv filter channels"}
    )
    aconv_filts: int = field(
        default=15, metadata={"help": "Number of attention conv filter size"}
    )
    cumulate_att_w: bool = field(
        default=True, metadata={"help": "Whether to cumulate previous attention weight"}
    )
    dlayers: int = field(
        default=2, metadata={"help": "Number of decoder lstm layers"}
    )
    dunits: int = field(
        default=1024, metadata={"help": "Number of decoder lstm units"}
    )
    prenet_layers: int = field(
        default=2, metadata={"help": "Number of prenet layers"}
    )
    prenet_units: int = field(
        default=256, metadata={"help": "Number of prenet units"}
    )
    postnet_layers: int = field(
        default=5, metadata={"help": "Number of postnet layers"}
    )
    postnet_filts: int = field(
        default=5, metadata={"help": "Number of postnet filter size"}
    )
    postnet_chans: int = field(
        default=512, metadata={"help": "Number of postnet filter channels"}
    )
    output_activation: str = field(
        default=None, metadata={"help": "Name of activation function for outputs"}
    )

    use_batch_norm: bool = field(
        default=True, metadata={"help": "Whether to use batch normalization"}
    )
    use_concate: bool = field(
        default=True, metadata={"help": "Whether to concat enc outputs w/ dec lstm outputs"}
    )
    use_residual: bool = field(
        default=False, metadata={"help": "Whether to use residual or not."}
    )
    reduction_factor: int = field(
        default=1, metadata={"help": "Reduction factor"}
    )
    spks: Optional[int] = field(
        default=None, metadata={
            "help": "Number of speakers, If set to > 1, assume that the sids will be provided as the input and use sid embedding layer"}
    )
    langs: Optional[int] = field(
        default=None, metadata={
            "help": "Number of languages If set to > 1, assume that the lids will be provided as the input and use sid embedding layer"}
    )
    spk_embed_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Speaker embedding dimension. If set to > 0, assume that spembs will be provided as the inpu"}
    )
    spk_embed_integration_type: str = field(
        default="concat", metadata={"help": "How to integrate speaker embedding"}
    )
    use_gst: bool = field(
        default=False, metadata={"help": "Whether to use global style token"}
    )
    gst_tokens: int = field(
        default=10, metadata={"help": "Number of GST embeddings"}
    )
    gst_heads: int = field(
        default=4, metadata={"help": "Number of heads in GST multihead attention"}
    )
    gst_conv_layers: int = field(
        default=6, metadata={"help": "Number of conv layers in GST"}
    )
    gst_conv_chans_list: List[int] = field(
        default=(32, 32, 64, 64, 128, 128), metadata={"help": "List of the number of channels of conv layers in GST"}
    )
    gst_conv_kernel_size: int = field(
        default=3, metadata={"help": "Kernel size of conv layers in GST"}
    )
    gst_conv_stride: int = field(
        default=2, metadata={"help": "Stride size of conv layers in GST"}
    )
    gst_gru_layers: int = field(
        default=1, metadata={"help": "Number of GRU layers in GST"}
    )
    gst_gru_units: int = field(
        default=128, metadata={"help": "Number of GRU units in GST"}
    )
    dropout_rate: float = field(
        default=0.5, metadata={"help": "Dropout rate"}
    )
    zoneout_rate: float = field(
        default=0.1, metadata={"help": "Zoneout rate"}
    )
    use_masking: bool = field(
        default=True, metadata={"help": "Whether to mask padded part in loss calculation"}
    )
    use_weighted_masking: bool = field(
        default=False, metadata={"help": "Whether to apply weighted masking in loss calculation"}
    )
    bce_pos_weight: float = field(
        default=5.0, metadata={"help": "Weight of positive sample of stop token only for use_masking=True"}
    )
    loss_type: str = field(
        default="L1+L2", metadata={"help": "Loss function type: L1, L2, or L1 + L2)"}
    )
    use_guided_attn_loss: bool = field(
        default=True, metadata={"help": "Whether to use guided attention loss"}
    )
    guided_attn_loss_sigma: float = field(
        default=0.4, metadata={"help": "Sigma in guided attention loss"}
    )
    guided_attn_loss_lambda: float = field(
        default=1.0, metadata={"help": "Lambda in guided attention loss"}
    )
    optimizer: str = field(
        default="adam", metadata={"help": "Optimizer for training."}
    )
