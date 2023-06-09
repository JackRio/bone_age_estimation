import os

import albumentations as A
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

from dataloader import BoneAgeDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using {} device".format(device))

# Define the transformations
transform = A.Compose([
    A.Resize(width=450, height=450),
    A.CenterCrop(width=250, height=250),
])

num_epochs = 100
batch_size = 16
learning_rate = 0.005

bad_train = BoneAgeDataset(annotations_file=os.path.join('data', 'rsna-bone-age', 'training', 'train_df.csv'),
                           transform=transform)
train_dataloader = DataLoader(bad_train, batch_size=batch_size, shuffle=True, num_workers=0)

bad_valid = BoneAgeDataset(annotations_file=os.path.join('data', 'rsna-bone-age', 'training', 'valid_df.csv'),
                           transform=transform)
valid_dataloader = DataLoader(bad_valid, batch_size=batch_size, shuffle=True, num_workers=0)

model = models.vgg16()
model.classifier[-1] = nn.Linear(in_features=4096, out_features=1)
model = model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)

# Train the model
total_step = len(train_dataloader)

for epoch in range(num_epochs):
    for i, scans in enumerate(train_dataloader):
        # Print the progress every 100 batches
        if (i + 1) % 100 == 0:
            print(f"\rEpoch {epoch + 1}/{num_epochs} [{i + 1}/{total_step}] ", end='')

        # Move tensors to the configured device
        images = scans['image'].to(torch.float32).to(device)
        labels = scans['boneage'].to(torch.float32).to(device)
        labels = torch.unsqueeze(labels, dim=1)

        # Forward pass
        outputs = model(images)

        loss = torch.sqrt(criterion(outputs, labels))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for scans in valid_dataloader:
            images = scans['image'].to(torch.float32).to(device)
            labels = scans['boneage'].to(torch.float32).to(device)

            with torch.no_grad():
                outputs = model(images)
                val_loss = criterion(outputs, labels)
            del images, labels, outputs

        print('Accuracy of the network on the {} validation images: {} %'.format(total, val_loss))

    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'output/model_vgg16_{}.ckpt'.format(epoch))
