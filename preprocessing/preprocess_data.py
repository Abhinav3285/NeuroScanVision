import cv2
import os
import numpy as np
from tqdm import tqdm

def preprocess_mri_images(input_dir, output_dir, size=(640, 640)):
    """
    Resizes and Normalizes MRI images for YOLOv8.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_name in tqdm(os.listdir(input_dir)):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            # 1. Load Image
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)

            # 2. Resize (Maintaining aspect ratio is best, but 640x640 is YOLO standard)
            img_resized = cv2.resize(img, size)

            # 3. Normalization (0-255 to 0-1) 
            # YOLOv8 does this internally, but we can ensure contrast is good
            img_normalized = cv2.normalize(img_resized, None, 0, 255, cv2.NORM_MINMAX)

            # 4. Save to processed folder
            cv2.imwrite(os.path.join(output_dir, img_name), img_normalized)

if __name__ == "__main__":
    # Example for Glioma folder
    preprocess_mri_images('../data/raw_images/glioma', '../data/yolo_dataset/images/train')
    print("Preprocessing complete!")