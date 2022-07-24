from dataclasses import dataclass, field
from typing import Optional, List, Dict

from deepaudio.tts.dataclass.configurations import DeepMMDataclass


@dataclass
class MelGANConfigs(DeepMMDataclass):
    name: str = field(
        default="melgan", metadata={"help": "Model name"}
    )
    in_channels: int = field(
        default=80, metadata={"help": "Number of input channels."}
    )
    out_channels: int = field(
        default=1, metadata={"help": "Number of output channels."}
    )
    kernel_size: int = field(
        default=7, metadata={"help": "Kernel size of initial and final conv layer."}
    )
    channels: int = field(
        default=512, metadata={"help": "Initial number of channels for conv layer."}
    )
    bias: bool = field(
        default=8, metadata={"help": "Whether to add bias parameter in convolution layers."}
    )
    upsample_scales: List[int] = field(
        default= [8, 8, 2, 2], metadata={"help": "List of upsampling scales."}
    )
    stack_kernel_size: int = field(
        default=3, metadata={"help": "Kernel size of dilated conv layers in residual stack.."}
    )
    stack: int = field(
        default=3, metadata={"help": "Number of stacks in a single residual stack.."}
    )
    nonlinear_activation: str = field(
        default="LeakyReLU", metadata={"help": "Activation function module name."}
    )
    nonlinear_activation_params: Dict = field(
        default={"negative_slope": 0.1}, metadata={"help": "Hyperparameters for activation function."}
    )
    pad: str = field(
        default="ReflectionPad1d", metadata={"help": "Padding function module name before dilated convolution layer."}
    )
    pad_params: Dict = field(
        default={}, metadata={"help": "Hyperparameters for padding function.."}
    )
    use_final_nonlinear_activation: bool = field(
        default=True, metadata={"help": "Activation function for the final layer.."}
    )
    use_weight_norm: bool = field(
        default=True, metadata={"help": "Whether to use weight norm in all of the conv layers."}
    )
    in_channels_discriminator: int = field(
        default=1, metadata={"help": "Number of input channels."}
    )
    out_channels_discriminator: int = field(
        default=1, metadata={"help": "Number of output channels."}
    )
    kernel_size_discriminator: int = field(
        default=7, metadata={"help": "Kernel size of initial and final conv layer."}
    )
    channels_discriminator: int = field(
        default=16, metadata={"help": "Initial number of channels for conv layer."}
    )
    max_downsample_channels_discriminator: int = field(
        default=1024, metadata={"help": "Maximum number of channels for downsampling layers."}
    )
    bias_discriminator: bool = field(
        default=8, metadata={"help": "Whether to add bias parameter in convolution layers."}
    )
    downsample_scales_discriminator: List[int] = field(
        default= [4, 4, 4, 4], metadata={"help": "List of upsampling scales."}
    )
    nonlinear_activation_discriminator: str = field(
        default="LeakyReLU", metadata={"help": "Activation function module name."}
    )
    nonlinear_activation_params_discriminator: Dict = field(
        default={"negative_slope": 0.1}, metadata={"help": "Hyperparameters for activation function."}
    )
    pad_discriminator: str = field(
        default="ReflectionPad1d", metadata={"help": "Padding function module name before dilated convolution layer."}
    )
    pad_params_discriminator: Dict = field(
        default={}, metadata={"help": "Hyperparameters for padding function.."}
    )
    use_final_nonlinear_activation_discriminator: bool = field(
        default=True, metadata={"help": "Activation function for the final layer.."}
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
    subband_stft_loss_params: Dict = field(
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

@dataclass
class StyleMelGANConfigs(DeepMMDataclass):
    name: str = field(
        default="style_melgan", metadata={"help": "Model name"}
    )
    in_channels: int = field(
        default=128, metadata={"help": "Number of input channels."}
    )
    aux_channels: int = field(
        default=80, metadata={"help": "Number of auxiliary input channels."}
    )
    channels: int = field(
        default=64, metadata={"help": "Number of channels for conv layer."}
    )
    out_channels: int = field(
        default=1, metadata={"help": "Number of output channels."}
    )
    kernel_size: int = field(
        default=7, metadata={"help": "Kernel size of initial and final conv layer."}
    )
    dilation: int = field(
        default=2, metadata={"help": "Dilation factor for conv layers."}
    )
    bias: bool = field(
        default=8, metadata={"help": "Whether to add bias parameter in convolution layers."}
    )
    noise_upsample_scales: List[int] = field(
        default=[8, 8, 2, 2], metadata={"help": "List of noise upsampling scales."}
    )
    noise_upsample_activation: str = field(
        default=[8, 8, 2, 2], metadata={"help": "Activation function module name for noise upsampling."}
    )
    noise_upsample_activation_params: Dict = field(
        default={"negative_slope": 0.1}, metadata={"help": "Hyperparameters for the above activation function."}
    )
    upsample_scales: List[int] = field(
        default= [8, 8, 2, 2], metadata={"help": "List of upsampling scales."}
    )
    upsample_mode: str = field(
        default="nearest", metadata={"help": "Upsampling mode in TADE layer."}
    )
    gated_function: str = field(
        default="softmax", metadata={"help": "Gated function used in TADEResBlock softmax or sigmoid."}
    )
    use_weight_norm: bool = field(
        default=True, metadata={"help": "Whether to use weight norm in all of the conv layers."}
    )
    repeats_discriminator: int = field(
        default=2, metadata={"help": "Number of repititons to apply RWD"}
    )
    window_size_discriminator: List[int] = field(
        default=[512, 1024, 2048, 4096], metadata={"help": "List of random window sizes."}
    )

    out_channels_discriminator: int = field(
        default=1, metadata={"help": "Number of output channels."}
    )
    kernel_size_discriminator: List[int] = field(
        default=[5, 3], metadata={"help": "Kernel size of initial and final conv layer."}
    )
    channels_discriminator: int = field(
        default=16, metadata={"help": "Initial number of channels for conv layer."}
    )
    max_downsample_channels_discriminator: int = field(
        default=512, metadata={"help": "Maximum number of channels for downsampling layers."}
    )
    bias_discriminator: bool = field(
        default=True, metadata={"help": "Whether to add bias parameter in convolution layers."}
    )
    downsample_scales_discriminator: List[int] = field(
        default= [4, 4, 4, 4], metadata={"help": "List of upsampling scales."}
    )
    nonlinear_activation_discriminator: str = field(
        default="LeakyReLU", metadata={"help": "Activation function module name."}
    )
    nonlinear_activation_params_discriminator: Dict = field(
        default={"negative_slope": 0.1}, metadata={"help": "Hyperparameters for activation function."}
    )
    pad_discriminator: str = field(
        default="ReflectionPad1d", metadata={"help": "Padding function module name before dilated convolution layer."}
    )
    pad_params_discriminator: Dict = field(
        default={}, metadata={"help": "Hyperparameters for padding function.."}
    )
    use_weight_norm_discriminator: bool = field(
        default=True, metadata={"help": "Whether to use weight norm in all of the conv layers."}
    )
    optimizer: str = field(
        default="adam", metadata={"help": "Optimizer for training."}
    )
