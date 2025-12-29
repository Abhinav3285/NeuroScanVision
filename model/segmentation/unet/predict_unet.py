import torch
import cv2
import numpy as np
from unet_model import UNet
from utils import preprocess_image

model = UNet()
model.load_state_dict(torch.load("model/segmentation/weights/unet.pt", map_location="cpu"))
model.eval()

def predict_mask(image_path):
    img = preprocess_image(image_path)
    with torch.no_grad():
        mask = model(img)
    return mask.numpy()
