import glob
import os

import cv2
import numpy as np
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

    @staticmethod
    def fill_mask(mask):
        # Fill the holes in the mask
        mask = mask.astype(np.uint8)
        filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
        return filled_mask

    def generate_mask(self, image, margin=25):
        self.predictor.set_image(image)
        input_point = np.array(
            [[image.shape[1] / 2, image.shape[0] / 2], [(image.shape[1] / 2) - 15, (image.shape[0] / 2) - 15]])
        input_label = np.array([1, 1])

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        best_mask = self.pick_best_mask(masks, scores)
        fill_mask = self.fill_mask(best_mask)
        cropped_image = self.crop_image_to_foreground(image, fill_mask, margin)
        return cropped_image, best_mask, fill_mask


if __name__ == "__main__":
    sam = SAM_Segmentation(sam_checkpoint="output/sam/sam_vit_h_4b8939.pth")
    images = glob.glob("data/sam_temp/original/*.png")
    for image in images:
        base_name = os.path.basename(image)
        image = cv2.imread(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        final_image, best_mask, fill_mask = sam.generate_mask(image)
        # save the cropped image
        os.makedirs("data/sam_temp/sam_output", exist_ok=True)
        cv2.imwrite(f"data/sam_temp/sam_output/{base_name}", final_image)
