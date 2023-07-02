import os

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from matplotlib.backends.backend_pdf import PdfPages

from config import constants as C
from models.model_zoo import BoneAgeEstModelZoo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using {} device".format(device))


def test_model(tc):
    transform = A.Compose([
        A.Resize(width=tc['image_size'], height=tc['image_size']),
        A.CLAHE(),
        A.Normalize(),
        ToTensorV2(),
    ])

    train_df = pd.read_csv(tc['test_df'])

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = tc["pretrained_filename"]
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = BoneAgeEstModelZoo(branch="gender", pretrained=False, lr=0.001).load_from_checkpoint(
            pretrained_filename)
        model.model.eval()
    else:
        print("No pretrained model found for testing")
        return

    # Create a PDF file
    with PdfPages(tc['pdf_filename']) as pdf:
        mean_error = []
        for row in train_df.iterrows():
            image = cv2.imread(row[1]['path'])
            processed_image = transform(image=image)['image']
            processed_image = processed_image.unsqueeze(0)
            processed_image = processed_image.to(device)
            boneage = torch.tensor(row[1]['boneage']).unsqueeze(0).unsqueeze(1).to(device)
            gender = torch.tensor(row[1]['gender']).unsqueeze(0).unsqueeze(1).to(device)
            scans = {
                'image': processed_image,
                'boneage': boneage,
                'gender': gender
            }

            val_result = model(scans)

            basename = os.path.basename(row[1]['path']).split('.')[0]
            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.set_title(
                f"Actual Age: {boneage.item()} months\nPredicted Age: {int(val_result.item())} months")
            ax.text(0.5, -0.1, f"Image: {basename}", transform=ax.transAxes, ha='center')

            ax.axis('off')
            mean_error.append(abs(boneage.item() - val_result.item()))
            # Save the figure to the PDF file
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    print("Mean error: ", np.mean(mean_error))

if __name__ == "__main__":
    # Load config file for training
    with open(C.MODEL_TEST_CONFIG, 'r') as stream:
        try:
            train_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        test_model(train_config)
