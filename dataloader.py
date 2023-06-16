import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class BoneAgeDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.img_labels = annotations_file
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.path.iloc[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gender = self.img_labels.gender.iloc[idx]
        boneage = self.img_labels.boneage.iloc[idx]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return {"image": image, "gender": gender[..., np.newaxis], "boneage": boneage[..., np.newaxis]}


if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    augmentations = A.Compose([
        # TODO: Check which size is better for cropping
        A.Resize(width=1024, height=1024),
        A.CenterCrop(width=768, height=768),
        A.CLAHE(),
        A.Normalize(),
        ToTensorV2(),
    ])

    bad_train = BoneAgeDataset(
        annotations_file=pd.read_csv(os.path.join('data', 'rsna-bone-age', 'training', 'train_df.csv')),
        transform=augmentations)

    dataloader = DataLoader(bad_train, batch_size=16, shuffle=False, num_workers=0)
    for i_batch, sample_batched in enumerate(dataloader):
        # plot an image from the batch
        print(sample_batched['image'][0].size())
        plt.imshow(sample_batched['image'][0].permute(1, 2, 0))
        plt.show()
        break
