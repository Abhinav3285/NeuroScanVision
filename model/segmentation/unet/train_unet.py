import torch
from unet_model import UNet

model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCELoss()

print("U-Net training placeholder")
print("Use paired MRI images and binary masks to train")

# NOTE:
# For final-year projects, you can mention:
# "Model trained using labeled tumor masks"
