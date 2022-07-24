from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from deepaudio.tts.dataclass.configurations import DeepMMDataclass


@dataclass
class VitsConfigs(DeepMMDataclass):
    name: str = field(
        default="vits", metadata={"help": "Model name"}
    )
    idim: int = field(
        default=1, metadata={"help": "Dimension of the inputs."}
    )
    odim: int = field(
        default=1, metadata={"help": "Dimension of the outputs."}
    )
    sampling_rate: int = field(
        default=22050, metadata={"help": "Sample rate."}
    )
    optimizer: str = field(
        default="adam", metadata={"help": "Optimizer for training."}
    )
