import pytorch_lightning as pl

from .config import settings
from .data import Logo2kDataModule
from .model import Logo2kTransferModel


def main():
    dm = Logo2kDataModule(settings.DATASET_FOLDER)
    model = Logo2kTransferModel(dm.num_classes)

    trainer = pl.Trainer(gpus=-1, max_epochs=1)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
