import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
from torch import nn, optim
from torchvision import models


class Logo2kTransferModel(pl.LightningModule):
    def __init__(self, output_classes: int, lr: float = 1e-4):
        super().__init__()

        self.save_hyperparameters()
        self.output_classes = output_classes
        self.lr = lr

        self.train_acc = torchmetrics.Accuracy()
        self.validation_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        # Load pretrained model (except for final layer), and freeze it
        backbone = models.resnet50(pretrained=True)
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False

        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Linear(num_filters, self.output_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.nll_loss(preds, y)

        self.train_acc(preds, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        self.log(
            "train_acc",
            self.train_acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.nll_loss(preds, y)

        self.validation_acc(preds, y)

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        self.log(
            "val_acc",
            self.validation_acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.nll_loss(preds, y)

        self.test_acc(preds, y)

        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        self.log(
            "test_acc",
            self.test_acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
