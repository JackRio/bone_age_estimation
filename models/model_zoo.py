import pytorch_lightning as L
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch import nn
from torchvision import models


class BoneAgeEstModelZoo(L.LightningModule):
    ARCHITECTURE = {
        "inception_v3": models.inception_v3,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet18": models.resnet18,
        "swin_b": models.swin_b,
    }
    BRANCH = ["medical", "gender", "swin_b"]

    def __init__(self, lr, architecture="inception_v3", branch="gender", pretrained=True):
        super().__init__()
        self.save_hyperparameters()
        self.loss_module = F.l1_loss
        self.lr = lr
        self.define_network()

    def define_network(self):
        if self.hparams.architecture not in self.ARCHITECTURE.keys():
            raise ValueError(f"Architecture must be one of {self.ARCHITECTURE.keys()}")
        if self.hparams.branch not in self.BRANCH:
            raise ValueError(f"Architecture must be one of {self.ARCHITECTURE}")
        self.model = self.ARCHITECTURE[self.hparams.architecture](pretrained=True)

        if self.hparams.branch == "medical" and self.hparams.architecture != "swin_b":
            self.medical = nn.Linear(17, 1)
            self.model.fc = nn.Linear(self.model.fc.in_features, 16)
        elif self.hparams.branch == "gender" and self.hparams.architecture != "swin_b":
            self.model.fc = nn.Linear(self.model.fc.in_features, 1000)
            self.gender = nn.Linear(1, 32)
            self.classifier = nn.Sequential(
                nn.Linear(self.model.fc.out_features + self.gender.out_features, 1000),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1000, 1000),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1000, 1),
            )

        if self.hparams.architecture == "inception_v3":
            self.model.aux_logits = False
        elif self.hparams.architecture == "swin_b":
            self.model.head = nn.Linear(self.model.head.in_features, 1)

        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, batch):
        x = self.model(batch['image'])
        if self.hparams.branch == "medical":
            gender = batch['gender'].to(torch.float32)
            z = torch.concat([x, gender], dim=1)
            return self.medical(z)
        elif self.hparams.branch == "gender":
            y = self.gender(batch['gender'].to(torch.float32))
            z = torch.cat((x, y), dim=1)
            return self.classifier(z)
        else:
            x = self.model(batch['image'])
            return x

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 25, 35, 45], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        preds = self(batch)

        loss = self.loss_module(preds, batch['boneage'])
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        with torch.no_grad():
            loss = self.loss_module(preds, batch['boneage'])
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_end(self):
        wandb.log({"val_loss_mean": self.trainer.callback_metrics["val_loss"].mean()})
