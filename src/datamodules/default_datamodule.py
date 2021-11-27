from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms 
from .datasets.handpose_dataset import HandposeDataset
from ..utils.data_utils import custom_split

class DefaultDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_val_split: tuple,
        batch_size: int = 32,
        transforms_train: Callable = torchvision.transforms.ToTensor(),
        transforms_train_pair: Optional[Callable] = None,
        transforms_test: Callable = torchvision.transforms.ToTensor(),
        num_workers: int = 4,
        pin_memory: bool = False,
        is_debugging: bool = False,
        **kwargs: int,
    ) -> None:

        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms_train = transforms_train
        self.transforms_train_pair = transforms_train_pair
        self.transforms_test = transforms_test

        self.data_train: Any = None
        self.data_val: Any = None
        self.data_test: Any = None

        self.is_debugging: bool = is_debugging

    def setup(self, stage: Any = None) -> None:
        # Transform and split datasets
        is_debugging = self.trainer.fast_dev_run or self.is_debugging
        trainset = HandposeDataset(
            self.data_dir, 
            train=True, 
            transforms=self.transforms_train, 
            transforms_pair=self.transforms_train_pair,
            limit = 20 if is_debugging else None
        )
        testset = HandposeDataset(
            self.data_dir, 
            train=False, 
            transforms=self.transforms_test, 
            limit = 20 if is_debugging else None
        )
        # We must use custom_split since images are repeated 4 times with different backgrounds,
        # and random split would leak data into the validation set. 
        self.data_train, self.data_val = custom_split(trainset, [10,10] if is_debugging else self.train_val_split)
        self.data_test = testset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        return None if self.train_val_split[1] == 0 else DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
