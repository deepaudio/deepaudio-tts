from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from deepaudio.tts.dataclass.configurations import DeepMMDataclass


@dataclass
class WaveRNNConfigs(DeepMMDataclass):
    name: str = field(
        default="wavernn", metadata={"help": "Model name"}
    )
    rnn_dims: int = field(
        default=1, metadata={"help": "Dims of FC Layers."}
    )
    bits: int = field(
        default=1, metadata={"help": "bit depth of signal."}
    )
    pad: int = field(
        default=22050, metadata={"help": "The context window size of the first convolution applied to the auxiliary "
                                         "input, by default 2"}
    )
    upsample_factors: List[int] = field(
        default=1, metadata={"help": "Upsample scales of the upsample network."}
    )
    feat_dims: int = field(
        default=22050, metadata={"help": "Auxiliary channel of the residual blocks."}
    )
    compute_dims: int = field(
        default=1, metadata={"help": "Dims of Conv1D in MelResNet."}
    )
    res_out_dims: int = field(
        default=1, metadata={"help": "Dims of output in MelResNet."}
    )
    res_blocks: int = field(
        default=22050, metadata={"help": "Number of residual blocks."}
    )
    mode: str = field(
        default=22050, metadata={"help": "Output mode of the WaveRNN vocoder.."}
    )
    optimizer: str = field(
        default="adam", metadata={"help": "Optimizer for training."}
    )