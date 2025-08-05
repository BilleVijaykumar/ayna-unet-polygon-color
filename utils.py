import os
import json
from PIL import Image
import torch
from torchvision import transforms

COLOR_DICT = {
    "red": [1, 0, 0],
    "green": [0, 1, 0],
    "blue": [0, 0, 1],
    "yellow": [1, 1, 0],
    "cyan": [0, 1, 1],
    "magenta": [1, 0, 1],
    "orange": [1, 0.5, 0],
    "purple": [0.5, 0, 0.5]
}

def load_data(json_path, input_dir, output_dir):
    with open(json_path) as f:
        data = json.load(f)

    samples = []
    for item in data:
        input_path = os.path.join(input_dir, item["input_polygon"])
        output_path = os.path.join(output_dir, item["output_image"])
        color = torch.tensor(COLOR_DICT[item["colour"]], dtype=torch.float32)
        samples.append((input_path, color, output_path))
    return samples

def get_transforms():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
