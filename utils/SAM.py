import glob
import os

import matplotlib.pyplot as plt
import pydicom
import cv2
import numpy as np
import pandas as pd
import torch
from segment_anything import sam_model_registry, SamPredictor


class SAM_Segmentation:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, sam_checkpoint, model_type="vit_h"):
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.predictor = self.load_model()

    def load_model(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        return SamPredictor(sam)

    @staticmethod
    def crop_image_to_foreground(image, binary_mask, margin):
        """
            Crop the image to the foreground object based on the binary mask.
        """

        # Set the background of main image using the binary mask
        image[binary_mask == 0] = 0

        margin = margin
        # Find the coordinates of the bounding box
        rows, cols = np.where(binary_mask == 1)
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)

        # Adjust the bounding box with the margin
        min_row = max(0, min_row - margin)
        max_row = min(binary_mask.shape[0] - 1, max_row + margin)
        min_col = max(0, min_col - margin)
        max_col = min(binary_mask.shape[1] - 1, max_col + margin)

        # Crop the original image based on the adjusted bounding box
        cropped_image = image[min_row:max_row + 1, min_col:max_col + 1]
        return cropped_image

    @staticmethod
    def pick_best_mask(masks, scores, weight_sum=0.7):
        # TODO: Maybe also use the score to pick the best mask

        sums = np.sum(masks, axis=(1, 2))
        # Find the index of the image with the highest sum
        combined_scores = [(s * weight_sum) * c for s, c in zip(sums, scores)]

        # Find the index of the number with the maximum combined score
        max_index = combined_scores.index(max(combined_scores))

        # Get the number with the maximum combined score
        best_mask = masks[max_index]
        return best_mask

    def manual_selection(self, masks):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].imshow(masks[0], cmap='Reds')
        axs[1].imshow(masks[1], cmap='Oranges')
        plt.tight_layout()
        plt.show()

        answer = input()
        return int(answer)

    @staticmethod
    def fill_mask(mask):
        # Fill the holes in the mask
        mask = mask.astype(np.uint8)
        filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        return filled_mask

    def get_largest_component(self, image):
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)

        # Find the largest component excluding the background (label 0)
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

        # Create a mask for the largest component
        largest_component_mask = np.uint8(labels == largest_label) * 1

        return largest_component_mask

    def generate_mask(self, image, margin=25):
        self.predictor.set_image(image)
        input_point = np.array(
            [
                [
                    image.shape[1] / 2, (image.shape[0] / 2)
                ],
                [
                    image.shape[1] / 2, (image.shape[0] / 2) + 150
                ],
                [
                    (image.shape[1] / 2) + 150, (image.shape[0] / 2)
                ]
            ]
        )
        input_label = np.array([1] * len(input_point))

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        # best_mask = self.pick_best_mask(masks, scores)
        best_mask_indx = self.manual_selection(masks[1:])
        if best_mask_indx == 3:
            return None
        fill_mask = self.fill_mask(masks[best_mask_indx])
        largest_connected_component = self.get_largest_component(fill_mask)
        cropped_image = self.crop_image_to_foreground(image, largest_connected_component, margin)
        return cropped_image


if __name__ == "__main__":
    dicom = True
    sam = SAM_Segmentation(sam_checkpoint="output/sam/sam_vit_h_4b8939.pth")
    rsna_raw = pd.read_csv("data/Mexico_private_dataset/mexico_additional_data.csv")
    # validation_images = glob.glob("data/rsna-bone-age/validation/boneage-validation-dataset-1/*")
    final_path = "data/Mexico_private_dataset/additional/"
    os.makedirs(final_path, exist_ok=True)
    for folder in rsna_raw.iterrows():
        image_path = folder[1]["path"]
        image_id = folder[1]["id"]
        if os.path.exists(f"data/Mexico_private_dataset/additional/{image_id}.png"):
            continue

        # read dicom image and convert to single channel
        if dicom:
            image = pydicom.dcmread(image_path).pixel_array
            # image to 255
            image = (image / np.max(image)) * 255
            rgb_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        else:
            image = cv2.imread(image_path)
            rgb_image = image

        final_image = sam.generate_mask(rgb_image)
        if final_image is None:
            continue

        # save the cropped image
        cv2.imwrite(f"data/Mexico_private_dataset/additional/{image_id}.png", final_image)
