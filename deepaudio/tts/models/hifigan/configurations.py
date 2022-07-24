from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from deepaudio.tts.dataclass.configurations import DeepMMDataclass


@dataclass
class HifiganConfigs(DeepMMDataclass):
    name: str = field(
        default="hifigan", metadata={"help": "Model name"}
    )
    in_channels: int = field(
        default=192, metadata={"help": "Number of input channels."}
    )
    out_channels: int = field(
        default=512, metadata={"help": "Number of output channels."}
    )
    channels: int = field(
        default=8, metadata={"help": "Number of hidden representation channels."}
    )
    global_channels: int = field(
        default=8, metadata={"help": "Number of global conditioning channels."}
    )
    kernel_size: int = field(
        default=8, metadata={"help": "Kernel size of initial and final conv layer."}
    )
    upsample_scales: List = field(
        default=8, metadata={"help": "List of upsampling scales."}
    )
    upsample_kernel_sizes: List = field(
        default=8, metadata={"help": "List of kernel sizes for upsampling layers."}
    )
    resblock_kernel_sizes: List = field(
        default=8, metadata={"help": "List of kernel sizes for residual blocks."}
    )
    resblock_dilations: List = field(
        default=8, metadata={"help": "List of dilation list for residual blocks."}
    )
    use_additional_convs: bool = field(
        default=8, metadata={"help": "Whether to use additional conv layers in residual blocks."}
    )
    bias: bool = field(
        default=8, metadata={"help": "Whether to add bias parameter in convolution layers."}
    )
    nonlinear_activation: str = field(
        default="LeakyReLU", metadata={"help": "Activation function module name."}
    )
    nonlinear_activation_params: Dict = field(
        default={"negative_slope": 0.1}, metadata={"help": "Hyperparameters for activation function."}
    )
    use_weight_norm: bool = field(
        default=True, metadata={"help": "Whether to use weight norm in all of the conv layers."}
    )
    scales: int = field(
        default=3, metadata={"help": "Number of multi-scales."}
    )
    scale_downsample_pooling: str = field(
        default="AvgPool1d", metadata={"help": "Pooling module name for downsampling of the inputs."}
    )
    scale_downsample_pooling_params: Dict[str, Any] = field(
        default={
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        }, metadata={"help": "Parameters for hifi-gan aboving pooling module."}
    )
    scale_discriminator_params: Dict[str, Any] = field(
        default={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        }, metadata={"help": "Parameters for hifi-gan scale discriminator module."}
    )
    follow_official_norm: bool = field(
        default=True, metadata={"help": "Whether to follow the norm setting of the official implementaion. The first "
                                        "discriminator uses spectral norm and the other discriminators use weight "
                                        "norm."}
    )
    periods: List[int] = field(
        default=[2, 3, 5, 7, 11], metadata={"help": "List of periods."}
    )
    period_discriminator_params: Dict[str, Any] = field(
        default={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        }, metadata={"help": "Parameters for hifi-gan period discriminator module. The period parameter will be "
                             "overwritten."}
    )
    optimizer: str = field(
        default="adam", metadata={"help": "Optimizer for training."}
    )