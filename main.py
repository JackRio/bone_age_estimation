import json
import os

import albumentations as A
import pandas as pd
import pytorch_lightning as L
import torch
import wandb
import yaml
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from config import constants as C
from dataloader import BoneAgeDataset
from models.model_zoo import BoneAgeEstModelZoo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using {} device".format(device))


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

    L.seed_everything(42)
    with open("data/wandb.json", "r") as f:
        wandb_config = json.load(f)
    wandb.login(key=wandb_config["wandb_api_key"], relogin=True)
    wandb.init(project=wandb_config["wandb_project"], entity=wandb_config["wandb_entity"], config=tc,
               name=tc["run_name"])
    wandb_logger = WandbLogger()

    trainer = L.Trainer(
        default_root_dir=os.path.join(tc["checkpoint_path"], tc["model_name"]),  # Where to save models
        precision=tc["precision"],
        accelerator="auto",
        devices=1,
        log_every_n_steps=4,
        max_epochs=tc["num_epochs"],
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=False, mode="min", monitor="val_loss", save_top_k=2
            ),
            LearningRateMonitor("epoch"),
        ],
    )
    pretrained_filename = tc["pretrained_filename"]
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = BoneAgeEstModelZoo(lr=tc["learning_rate"], architecture=tc["model_name"], branch=tc["branch"],
                                   pretrained=tc["pretrained"])
    else:
        model = BoneAgeEstModelZoo(lr=tc["learning_rate"], architecture=tc["model_name"], branch=tc["branch"],
                                   pretrained=tc["pretrained"])
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
