import json
import os

import albumentations as A
import pandas as pd
import pytorch_lightning as L
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
import yaml
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from config import constants as C
from dataloader import BoneAgeDataset


class Inception(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.loss_module = F.l1_loss
        self.lr = lr
        self.model = models.inception_v3(pretrained=True)
        self.model.aux_logits = False

        self.gender_branch = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1)
        )

    def forward(self, batch):
        x = self.model(batch['image'])
        gender = batch['gender'].to(torch.float32)
        y = self.gender_branch(gender)
        z = torch.cat((x, y), dim=1)
        z = self.classifier(z)
        return z

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1, )
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


def train_model(tc):
    transform = A.Compose([
        A.Resize(width=tc['image_size'], height=tc['image_size']),
        A.Flip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(),
        A.Normalize(),
        ToTensorV2(),
    ])
    raw_train_df = pd.read_csv(tc["train_df"])
    valid_df = pd.read_csv(tc["valid_df"])

    bad_train = BoneAgeDataset(annotations_file=raw_train_df, transform=transform)
    train_loader = DataLoader(bad_train, batch_size=tc["batch_size"], shuffle=True, num_workers=0)

    bad_valid = BoneAgeDataset(annotations_file=valid_df, transform=transform)
    val_loader = DataLoader(bad_valid, batch_size=tc["batch_size"], shuffle=False, num_workers=0)
    with open("data/wandb.json", "r") as f:
        wandb_config = json.load(f)
    wandb.login(key=wandb_config["wandb_api_key"], relogin=True)
    wandb.init(project=wandb_config["wandb_project"], entity=wandb_config["wandb_entity"], config=tc,
               name=tc["run_name"])
    wandb_logger = WandbLogger()

    L.seed_everything(42)

    trainer = L.Trainer(
        default_root_dir=os.path.join(tc["checkpoint_path"], tc["model_name"]),  # Where to save models
        precision=tc["precision"],
        accelerator="auto",
        devices=1,
        max_epochs=tc["num_epochs"],
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="min", monitor="val_loss", every_n_epochs=5, save_top_k=2
            ),
            LearningRateMonitor("epoch"),
        ],
    )
    pretrained_filename = os.path.join(tc["pretrained_filename"])
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = Inception.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(42)
        model = Inception(lr=tc["learning_rate"])
    print(model)
    trainer.fit(model, train_loader, val_loader)
    return model, trainer


if __name__ == "__main__":
    # Load config file for training
    with open(C.MODEL_TRAIN_CONFIG, 'r') as stream:
        try:
            train_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    final_model, final_result = train_model(train_config)
    print(final_result)
