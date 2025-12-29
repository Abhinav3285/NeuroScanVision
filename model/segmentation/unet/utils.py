import cv2
import torch
import numpy as np

def preprocess_image(img_path, size=256):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img

def overlay_mask(image, mask):
    mask = mask.squeeze()
    mask = (mask > 0.5).astype(np.uint8) * 255
    colored = image.copy()
    colored[:, :, 0] = np.maximum(colored[:, :, 0], mask)
    return colored
