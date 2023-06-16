import os

import albumentations as A
import pandas as pd
import pytorch_lightning as L
import torch
import json
import wandb
import yaml
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from config import constants as C
from dataloader import BoneAgeDataset
from models.resnet import ResNet
from pytorch_lightning.loggers import WandbLogger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using {} device".format(device))

transform = A.Compose([
    A.Resize(width=512, height=512),
    A.CLAHE(),
    A.Normalize(),
    ToTensorV2(),
])


def train_model(tc):
    raw_train_df = pd.read_csv(tc["train_df"])
    valid_df = pd.read_csv(tc["valid_df"])

    bad_train = BoneAgeDataset(annotations_file=raw_train_df, transform=transform)
    train_loader = DataLoader(bad_train, batch_size=tc["batch_size"], shuffle=True, num_workers=0)

    bad_valid = BoneAgeDataset(annotations_file=valid_df, transform=transform)
    val_loader = DataLoader(bad_valid, batch_size=tc["batch_size"], shuffle=False, num_workers=0)

    with open("data/wandb.json", "r") as f:
        wandb_config = json.load(f)
    wandb.login(key=wandb_config["wandb_api_key"], relogin=True)
    wandb.init(project=wandb_config["wandb_project"], entity=wandb_config["wandb_entity"], config=tc)
    wandb_logger = WandbLogger()

    trainer = L.Trainer(
        default_root_dir=os.path.join(tc["checkpoint_path"], tc["model_name"]),  # Where to save models
        precision=tc["precision"],
        accelerator="auto",
        devices=1,
        max_epochs=tc["num_epochs"],
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="val_acc"
            ),
            LearningRateMonitor("epoch"),
        ],
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(tc["pretrained_filename"])
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = ResNet.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(42)  # To be reproducable

        # TODO: Change this class name to load the appropriate model
        model = ResNet(resent_version=tc["model_name"], pretrained=True, lr=tc['learning_rate'])
        trainer.fit(model, train_loader, val_loader)
        model = ResNet.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    result = {
        "val": val_result[0]["test_acc"]
    }

    return model, result


if __name__ == "__main__":
    # Load config file for training
    with open(C.MODEL_TRAIN_CONFIG, 'r') as stream:
        try:
            train_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    final_model, final_result = train_model(train_config)
    print(final_result)
