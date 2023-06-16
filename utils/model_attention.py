import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import pandas as pd
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader

from config import constants as C
from dataloader import BoneAgeDataset
from models.resnet import ResNet

with open(C.MODEL_ATTENTION_CONFIG, "r") as f:
    config = yaml.safe_load(f)

augmentations = A.Compose([
    A.Resize(width=512, height=512),
    A.Normalize(),
    ToTensorV2()
])
dataset = BoneAgeDataset(pd.read_csv(config['annotation_file']), transform=augmentations)
val_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
resnet = ResNet.load_from_checkpoint('lightning_logs/fmt8pml9/checkpoints/epoch=0-step=148.ckpt')

for i, scan in enumerate(val_loader):
    input_tensor = scan['image']

    target_layers = [resnet.model.layer4[-1]]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=resnet.model, target_layers=target_layers, use_cuda=True)

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(input_tensor.numpy(), grayscale_cam, use_rgb=True)

print(visualization)
