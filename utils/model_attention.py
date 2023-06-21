import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as transforms
import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.resnet import ResNet

resnet = ResNet.load_from_checkpoint('lightning_logs/mkmcrxik/checkpoints/epoch=14-step=4440.ckpt')
target_layers = [resnet.model.layer4[-1]]
resnet.model.eval()

validation_df = pd.read_csv("data/rsna-bone-age/training/valid_df.csv").head(300)
ids = validation_df["id"].tolist()

# Create a PDF file for saving plots
pdf_path = "output/resnet_pretrained_validation.pdf"
with PdfPages(pdf_path) as pdf:
    for id in tqdm.tqdm(ids):
        image = cv2.imread(validation_df[validation_df["id"] == id]["path"].tolist()[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Define the transformations
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        # Apply the transformations to the image
        tensor_image = transform(image).unsqueeze(0)
        cuda_tensor = tensor_image.to(device='cuda')
        output = resnet.model(cuda_tensor)
        predicted = int(output.item())
        actual = validation_df[validation_df["id"] == id]["boneage"].tolist()[0]
        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=resnet.model, target_layers=target_layers, use_cuda=True)

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=tensor_image)
        visualization = show_cam_on_image(tensor_image[0].permute(1, 2, 0).numpy(), grayscale_cam[0], use_rgb=True)

        # Add colorbar for the heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the visualization with colorbar
        img = ax1.imshow(visualization)
        fig.colorbar(img, ax=ax1)

        # Plot the tensor_image
        ax2.imshow(tensor_image[0].permute(1, 2, 0).numpy())
        ax2.axis('off')

        # Add text with output and predicted values
        text = f"Predicted: {predicted} | Actual: {actual}"
        plt.text(10, 30, text, fontsize=12, color='white', backgroundcolor='black')
        plt.title(f"ID: {id}")

        # Save the plot to the PDF file
        pdf.savefig()

        # Close the plot to free memory
        plt.close()
