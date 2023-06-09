import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class BoneAgeDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
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
            image = augmented["image"].transpose(2, 0, 1)
        return {"image": image, "gender": gender, "boneage": boneage}


if __name__ == "__main__":
    import albumentations as A

    transform = A.Compose([
        A.Resize(width=450, height=450),
        A.CenterCrop(width=350, height=350),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
    ])

    bad_train = BoneAgeDataset(annotations_file=os.path.join('data', 'rsna-bone-age', 'training', 'train_df.csv'),
                               transform=transform)

    dataloader = DataLoader(bad_train, batch_size=16, shuffle=True, num_workers=0)
    print(len(dataloader))
    for i_batch, sample_batched in enumerate(dataloader):
        # plot an image from the batch
        print(sample_batched['image'][0].size())
        plt.imshow(sample_batched['image'][0].permute(1, 2, 0))
        plt.show()
        break

