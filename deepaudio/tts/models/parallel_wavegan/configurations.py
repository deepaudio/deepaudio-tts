from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from deepaudio.tts.dataclass.configurations import DeepMMDataclass


@dataclass
class ParallelWaveganConfigs(DeepMMDataclass):
    name: str = field(
        default="parallel_wavegan", metadata={"help": "Model name"}
    )
    in_channels: int = field(
        default=1, metadata={"help": "Number of input channels."}
    )
    out_channels: int = field(
        default=1, metadata={"help": "Number of output channels."}
    )
    kernel_size: int = field(
        default=3, metadata={"help": "Kernel size of initial and final conv layer."}
    )
    layers: int = field(
        default=10, metadata={"help": "Number of conv layers."}
    )
    stacks: int = field(
        default=3, metadata={"help": "Number of stacks i.e., dilation cycles."}
    )
    residual_channels: int = field(
        default=64, metadata={"help": "Number of channels in residual conv."}
    )
    gate_channels: int = field(
        default=128, metadata={"help": "Number of channels in gated conv."}
    )
    skip_channels: int = field(
        default=64, metadata={"help": "Number of channels in skip conv."}
    )
    aux_channels: int = field(
        default=80, metadata={"help": "Number of channels for auxiliary feature conv."}
    )
    dropout_rate: float = field(
        default=0.0, metadata={"help": "Dropout rate. 0.0 means no dropout applied."}
    )
    aux_context_window: int = field(
        default=2, metadata={"help": "Context window size for auxiliary feature."}
    )
    bias: bool = field(
        default=True, metadata={"help": "Whether to use bias parameter in conv."}
    )
    use_weight_norm: bool = field(
        default=True, metadata={"help": "Whether to use weight norm in all of the conv layers."}
    )
    upsample_conditional_features: bool = field(
        default=True, metadata={"help": "Whether to use upsampling network."}
    )
    upsample_net: str = field(
        default="ConvInUpsampleNetwork", metadata={"help": "Upsampling network architecture."}
    )
    upsample_params: Dict[str, Any] = field(
        default={"upsample_scales": [4, 4, 4, 4]}, metadata={"help": "Upsampling network parameters."}
    )
    in_channels_discriminator: int = field(
        default=1, metadata={"help": " Number of input channels."}
    )
    out_channels_discriminator: int = field(
        default=1, metadata={"help": " Number of output channels."}
    )
    kernel_size_discriminator: int = field(
        default=3, metadata={"help": "Kernel size of initial and final conv layer."}
    )
    layers_discriminator: int = field(
        default=10, metadata={"help": "Number of conv layers."}
    )
    conv_channels_discriminator: int = field(
        default=64, metadata={"help": "Number of chnn layers."}
    )
    dilation_factor_discriminator: int = field(
        default= 1, metadata={"help": "Dilation factor."}
    )
    nonlinear_activation_discriminator: str = field(
        default="LeakyReLU", metadata={"help": "Activation function module name."}
    )
    nonlinear_activation_params_discriminator: Dict = field(
        default={"negative_slope": 0.1}, metadata={"help": "Hyperparameters for activation function."}
    )
    bias_discriminator: bool = field(
        default=True, metadata={"help": "Whether to use bias parameter in conv."}
    )
    use_weight_norm_discriminator: bool = field(
        default=True, metadata={"help": "Whether to use weight norm in all of the conv layers."}
    )
    stft_loss_params: Dict = field(
        default={
            'fft_sizes': [1024, 2048, 512],
            'hop_sizes': [120, 240, 50],
            'win_lengths': [600, 1200, 240],
            'window': "hann"
        }, metadata={"help": "MultiResolutionSTFTLoss parameters."}
    )

    optimizer: str = field(
        default="adam", metadata={"help": "Optimizer for training."}
    )
