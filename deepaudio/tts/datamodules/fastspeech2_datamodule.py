from typing import Any, Dict, Optional, Tuple
import jsonlines
import numpy as np

from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from deepaudio.tts.datasets.am_batch_fn import fastspeech2_multi_spk_batch_fn
from deepaudio.tts.datasets.am_batch_fn import fastspeech2_single_spk_batch_fn
from deepaudio.tts.datasets.data_table import DataTable


class Fastspeech2DataModule(LightningDataModule):
    def __init__(self,
                 train_metadata: str,
                 dev_metadata: str,
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 speaker_dict: Optional[str] = None,
                 voice_cloning: Optional[bool] = False,
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_dataset: Optional[Dataset] = None
        self.dev_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        fields = [
            "text", "text_lengths", "speech", "speech_lengths", "durations",
            "pitch", "energy"
        ]
        converters = {"speech": np.load, "pitch": np.load, "energy": np.load}
        spk_num = None
        if self.hparams.speaker_dict is not None:
            print("multiple speaker fastspeech2!")
            self.collate_fn = fastspeech2_multi_spk_batch_fn
            with open(self.hparams.speaker_dict, 'rt') as f:
                spk_id = [line.strip().split() for line in f.readlines()]
            spk_num = len(spk_id)
            fields += ["spk_id"]
        elif self.hparams.voice_cloning:
            print("Training voice cloning!")
            self.collate_fn = fastspeech2_multi_spk_batch_fn
            fields += ["spk_emb"]
            converters["spk_emb"] = np.load
        else:
            print("single speaker fastspeech2!")
            self.collate_fn = fastspeech2_single_spk_batch_fn
        print("spk_num:", spk_num)

        # construct dataset for training and validation
        with jsonlines.open(self.hparams.train_metadata, 'r') as reader:
            train_metadata = list(reader)
        self.train_dataset = DataTable(
            data=train_metadata,
            fields=fields,
            converters=converters, )
        with jsonlines.open(self.hparams.dev_metadata, 'r') as reader:
            dev_metadata = list(reader)
        self.dev_dataset = DataTable(
            data=dev_metadata,
            fields=fields,
            converters=converters, )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dev_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
