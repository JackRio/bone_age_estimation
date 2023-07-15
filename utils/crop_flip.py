import os
import cv2
import glob
import matplotlib.pyplot as plt

paths = glob.glob("data/Mexico_private_dataset/preprocessed/*")
for path in paths:
    if os.path.exists(os.path.join("data/Mexico_private_dataset/reiterate/", os.path.basename(path))):
        continue
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    plt.imshow(img)
    plt.show()
    y_crop = input()
    box = (0, 0, int(y_crop), img.shape[1])
    cropped = img[box[0]:box[2], box[1]:box[3]]
    cv2.imwrite(os.path.join("data/Mexico_private_dataset/reiterate/", os.path.basename(path)), cropped)
