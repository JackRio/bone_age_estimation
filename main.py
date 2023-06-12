import os

import albumentations as A
import pandas as pd
import pytorch_lightning as L
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from config import constants as C
from dataloader import BoneAgeDataset
from models.swin_b import SwinB

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using {} device".format(device))
# Define the transformations
transform = A.Compose([
    A.Resize(width=450, height=450),
    A.CenterCrop(width=256, height=256),
    A.Normalize(),
    ToTensorV2(),
])

num_epochs = 50
batch_size = 16
learning_rate = 0.0005

# Load the data
raw_train_df = pd.read_csv(C.TRAIN_CSV)
valid_df = pd.read_csv(C.VALID_CSV)

bad_train = BoneAgeDataset(annotations_file=raw_train_df, transform=transform)
train_dataloader = DataLoader(bad_train, batch_size=batch_size, shuffle=True, num_workers=0)

bad_valid = BoneAgeDataset(annotations_file=valid_df, transform=transform)
valid_dataloader = DataLoader(bad_valid, batch_size=batch_size, shuffle=False, num_workers=0)


def train_model(**kwargs):
    trainer = L.Trainer(
        default_root_dir=os.path.join(C.CHECKPOINT_PATH, "SwinB"),
        accelerator="auto",
        devices=1,
        max_epochs=50,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
            LearningRateMonitor("epoch"),
        ]
        # logger=L.loggers.TensorBoardLogger(save_dir=C.LOGS_PATH, name="swinb")
    )
    # trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    # trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    print(trainer.checkpoint_callback.best_model_path)

    if os.path.isfile(C.PRETRAINED_FILENAME):
        print("Found pretrained model at %s, loading..." % C.PRETRAINED_FILENAME)
        # Automatically loads the model with the saved hyperparameters
        if C.PRETRAINED_FILENAME.endswith("ckpt"):
            model = SwinB.load_from_checkpoint(C.PRETRAINED_FILENAME)
        else:
            model = SwinB(**kwargs)
            model.model.load_state_dict(torch.load(C.PRETRAINED_FILENAME))
            model.model.head = nn.Linear(in_features=1024, out_features=1)
        if is_training:
            trainer.fit(model, train_dataloader, valid_dataloader)
            model = SwinB.load_state_dict(trainer.checkpoint_callback.best_model_path)
    else:
        L.seed_everything(42)  # To be reproducable
        model = SwinB(**kwargs)
        trainer.fit(model, train_dataloader, valid_dataloader)
        # Load best checkpoint after training
        model = SwinB.load_state_dict(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=valid_dataloader, verbose=False)
    #     test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {
        #         "test": test_result[0]["test_acc"],
        "val": val_result[0]["test_acc"]
    }

    return model, result


is_training = True
train_model(model_kwargs={}, lr=0.0001)
