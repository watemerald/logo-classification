from typing import Any, Callable, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class Logo2kDataModule(pl.LightningDataModule):
    name = "Logo2k+"
    dims = (3, 256, 256)

    def __init__(
        self,
        data_dir: str,
        train_val_test_split: Union[
            Tuple[float, float, float], Tuple[int, int, int]
        ] = (0.7, 0.2, 0.1),
        num_workers: int = 16,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle

    @property
    def num_classes(self) -> int:
        return 10

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = ImageFolder(self.data_dir)
        splits = self._get_splits(dataset)
        train, val, test = random_split(
            dataset, splits, generator=torch.Generator().manual_seed(self.seed)
        )
        if stage == "fit" or stage is None:
            self.train_dataset = SubsetTransform(
                train,
                self.test_transforms,
            )
            self.val_dataset = SubsetTransform(val, self.test_transforms)

        if stage == "test" or stage is None:
            self.test_dataset = SubsetTransform(test, self.test_transforms)

    def train_dataloader(self):
        return self._data_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._data_loader(self.val_dataset)

    def test_dataloader(self):
        return self._data_loader(self.test_dataset)

    @property
    def test_transforms(self) -> Callable:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def _get_splits(self, dataset: Dataset) -> List[int]:
        train, val, test = self.train_val_test_split
        if isinstance(train, int):
            train, val, test = self.train_val_test_split
            return [train, val, test]  # type: ignore
        elif isinstance(train, float):
            dataset_len = len(dataset)  # type: ignore[arg-type]
            train_len = int(train * dataset_len)
            val_len = int(test * dataset_len)
            test_len = dataset_len - train_len - val_len

            return [train_len, val_len, test_len]
        else:
            raise ValueError(f"Unsupported type {type(self.train_val_test_split[0])}")

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )


class SubsetTransform(Dataset):
    def __init__(self, subset: Subset, transforms: Optional[Callable] = None):
        self.subset = subset
        self.transforms = transforms

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transforms:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return len(self.subset)
