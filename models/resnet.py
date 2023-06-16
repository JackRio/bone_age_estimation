import pytorch_lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import torchvision.models as models


class ResNet(L.LightningModule):
    version = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }

    def __init__(self, resent_version, pretrained, lr):
        super().__init__()
        self.save_hyperparameters()
        print(f"Using {resent_version} model")
        self.model = self.version[resent_version](pretrained=pretrained)

        linear_in_features = list(self.model.children())[-1].in_features
        self.model.fc = nn.Linear(linear_in_features, 1)

        # TODO: Freeze the layers excpet FC

    def forwards(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15], gamma=0.1, )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        preds = self.model(batch['image'])
        loss = F.l1_loss(preds, batch['boneage'])

        self.log("%s_loss" % mode, loss, prog_bar=True)
        self.log("%s_acc" % mode, loss, prog_bar=True)
        wandb.log({"%s/loss" % mode: loss})
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
