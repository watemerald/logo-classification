import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from .config import settings
from .data import Logo2kDataModule
from .model import Logo2kTransferModel


def main():
    dm = Logo2kDataModule(settings.DATASET_FOLDER)
    model = Logo2kTransferModel(dm.num_classes)

    wandb_logger = WandbLogger(project="logo-classification")
    tensorboard_logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=10,
        logger=[wandb_logger, tensorboard_logger],
        callbacks=[EarlyStopping(monitor="val_acc", patience=2)],
    )
    trainer.fit(model, dm)

    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
