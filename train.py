import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import wandb
from model.unet import UNet
from utils import load_data, get_transforms

class PolygonDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, color, output_path = self.samples[idx]
        input_img = Image.open(input_path).convert("L")
        output_img = Image.open(output_path).convert("RGB")
        return self.transform(input_img), color, self.transform(output_img)

def train():
    wandb.init(project="polygon-color-unet")
    train_data = load_data("dataset/training/data.json", "dataset/training/inputs", "dataset/training/outputs")
    val_data = load_data("dataset/validation/data.json", "dataset/validation/inputs", "dataset/validation/outputs")

    transform = get_transforms()
    train_loader = DataLoader(PolygonDataset(train_data, transform), batch_size=16, shuffle=True)
    val_loader = DataLoader(PolygonDataset(val_data, transform), batch_size=16)

    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(20):
        model.train()
        total_loss = 0
        for x, cond, y in train_loader:
            x, cond, y = x.to(device), cond.to(device), y.to(device)
            pred = model(x, cond)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        wandb.log({"train_loss": total_loss / len(train_loader)})
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    # ✅ Save the model after training
    torch.save(model.state_dict(), "unet_model.pth")
    print("Model saved as unet_model.pth ✅")


if __name__ == "__main__":
    train()
