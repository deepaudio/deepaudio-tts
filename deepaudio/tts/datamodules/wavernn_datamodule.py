from typing import Any, Dict, Optional, Tuple
import jsonlines
import numpy as np

from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from deepaudio.tts.datasets.vocoder_batch_fn import WaveRNNClip
from deepaudio.tts.datasets.data_table import DataTable


class WaveRNNDataModule(LightningDataModule):
    def __init__(self,
                 train_metadata: str,
                 dev_metadata: str,
                 batch_max_steps: int,
                 n_shift: int,
                 mode: str,
                 bits: int,
                 aux_context_window: Optional[int] = 0,
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
            fields=["wave", "feats"],
            converters={
                "wave": np.load,
                "feats": np.load,
            }, )

        with jsonlines.open(self.hparams.dev_metadata, 'r') as reader:
            dev_metadata = list(reader)
        self.dev_dataset = DataTable(
            data=dev_metadata,
            fields=["wave", "feats"],
            converters={
                "wave": np.load,
                "feats": np.load,
            }, )

        self.collate_fn = WaveRNNClip(
            mode=self.hparams.mode,
            aux_context_window=self.hparams.aux_context_window,
            hop_size=self.hparams.n_shift,
            batch_max_steps=self.hparams.batch_max_steps,
            bits=self.hparams.bits)

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
