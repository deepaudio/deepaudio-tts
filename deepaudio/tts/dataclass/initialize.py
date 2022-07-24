from hydra.core.config_store import ConfigStore


from .configurations import (
    CPUTrainerConfigs,
    GPUTrainerConfigs,
    TPUTrainerConfigs,
    Fp16GPUTrainerConfigs,
    Fp16TPUTrainerConfigs,
    Fp64CPUTrainerConfigs,
)


SPEAKER_TRAIN_CONFIGS = [
    "feature",
    "augment",
    "dataset",
    "model",
    "criterion",
    "lr_scheduler",
    "trainer",
]


TRAINER_DATACLASS_REGISTRY = {
    "cpu": CPUTrainerConfigs,
    "gpu": GPUTrainerConfigs,
    "tpu": TPUTrainerConfigs,
    "gpu-fp16": Fp16GPUTrainerConfigs,
    "tpu-fp16": Fp16TPUTrainerConfigs,
    "cpu-fp64": Fp64CPUTrainerConfigs,
}


def hydra_train_init() -> None:
    r""" initialize ConfigStore for hydra-train """
    from deepaudio.speaker.models import MODEL_DATACLASS_REGISTRY
    from deepaudio.speaker.optim.scheduler import SCHEDULER_DATACLASS_REGISTRY

    registries = {
        "trainer": TRAINER_DATACLASS_REGISTRY,
        "model": MODEL_DATACLASS_REGISTRY,
        "lr_scheduler": SCHEDULER_DATACLASS_REGISTRY,
    }

    cs = ConfigStore.instance()

    for group in SPEAKER_TRAIN_CONFIGS:
        dataclass_registry = registries[group]

        for k, v in dataclass_registry.items():
            cs.store(group=group, name=k, node=v, provider="deepaudio")

