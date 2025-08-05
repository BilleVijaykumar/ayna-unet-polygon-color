# inference.py

import torch
from PIL import Image
import matplotlib.pyplot as plt
from model.unet import UNet
from utils import get_transforms, COLOR_DICT

# --- Load the model ---
model = UNet()
model.load_state_dict(torch.load("unet_model.pth", map_location=torch.device('cpu')))
model.eval()

# --- Load input image ---
img = Image.open("dataset/validation/inputs/octagon.png").convert("L")  # ✅ Check path
color_name = "red"  # 🔴 You can change to 'blue', 'green', etc.

# --- Convert color name to tensor ---
cond = torch.tensor(COLOR_DICT[color_name], dtype=torch.float).unsqueeze(0)  # ✅ Must be float

# --- Transform input ---
transform = get_transforms()
x = transform(img).unsqueeze(0)

# --- Inference ---
with torch.no_grad():
    pred = model(x, cond).squeeze(0).permute(1, 2, 0).numpy()

# --- Plot input and output ---
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Input Polygon")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pred.clip(0, 1))
plt.title(f"Colorized: {color_name}")
plt.axis("off")


plt.tight_layout()
plt.show()
