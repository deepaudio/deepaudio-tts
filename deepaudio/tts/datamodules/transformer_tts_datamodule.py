from typing import Any, Dict, Optional, Tuple
import jsonlines
import numpy as np

from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from deepaudio.tts.datasets.am_batch_fn import transformer_single_spk_batch_fn
from deepaudio.tts.datasets.data_table import DataTable


class TransformerTTSDataModule(LightningDataModule):
    def __init__(self,
                 train_metadata: str,
                 dev_metadata: str,
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_dataset: Optional[Dataset] = None
        self.dev_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # construct dataset for training and validation
        with jsonlines.open(self.hparams.train_metadata, 'r') as reader:
            train_metadata = list(reader)
        self.train_dataset = DataTable(
            data=train_metadata,
            fields=[
                "text",
                "text_lengths",
                "speech",
                "speech_lengths",
            ],
            converters={
                "speech": np.load,
            }, )
        with jsonlines.open(self.hparams.dev_metadata, 'r') as reader:
            dev_metadata = list(reader)
        self.dev_dataset = DataTable(
            data=dev_metadata,
            fields=[
                "text",
                "text_lengths",
                "speech",
                "speech_lengths",
            ],
            converters={
                "speech": np.load,
            }, )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=transformer_single_spk_batch_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dev_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=transformer_single_spk_batch_fn,
        )
