import os

import albumentations as A
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader

from config import constants as C
from dataloader import BoneAgeDataset
from models.resnet import ResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using {} device".format(device))


def test_model(tc):
    transform = A.Compose([
        A.Resize(width=tc['image_size'], height=tc['image_size']),
        A.Normalize(),
        ToTensorV2(),
    ])
    valid_df = pd.read_csv(tc["valid_df"])

    bad_valid = BoneAgeDataset(annotations_file=valid_df, transform=transform)
    val_loader = DataLoader(bad_valid, batch_size=tc["batch_size"], shuffle=False, num_workers=0)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(tc["pretrained_filename"])
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = ResNet.load_from_checkpoint(pretrained_filename)
        model.model.eval()
    else:
        print("No pretrained model found for testing")
        return

    # Create a PDF file
    with PdfPages(tc['pdf_filename']) as pdf:
        for i, scans in enumerate(val_loader):
            val_result = model(scans['image'].to(device))

            # Save input images with actual and predicted bone ages
            for j in range(len(scans['image'])):
                image = scans['image'][j]
                actual_age = scans['boneage'][j]
                predicted_age = val_result[j]

                fig, ax = plt.subplots()
                ax.imshow(image.permute(1, 2, 0))
                ax.set_title(
                    f"Actual Age: {actual_age.item()} months\nPredicted Age: {int(predicted_age.item())} months")
                ax.axis('off')

                # Save the figure to the PDF file
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()


if __name__ == "__main__":
    # Load config file for training
    with open(C.MODEL_TEST_CONFIG, 'r') as stream:
        try:
            train_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        test_model(train_config)
