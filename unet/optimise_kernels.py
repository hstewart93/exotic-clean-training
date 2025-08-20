import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import optuna
import json

from unet.models import UNet


def train_and_evaluate(model, train_loader, val_loader, epochs=20, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = next(model.parameters()).device

    for epoch in range(epochs):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, masks).item() * images.size(0)
    val_loss /= len(val_loader.dataset)

    return val_loss


def objective(trial):
    # Suggest kernel sizes (odd numbers between 3 and 17)
    kernel_sizes = []
    for i in range(num_layers):
        kH = trial.suggest_int(f"kernel_height_{i}", 3, 17, step=2)
        kW = trial.suggest_int(f"kernel_width_{i}", 3, 17, step=2)
        kernel_sizes.append((kH, kW))

    # Update model config with suggested kernel sizes
    model_config = default_config.copy()
    model_config["kernel_sizes"] = kernel_sizes

    # Initialize your model with the trial kernel sizes
    model = UNet(**model_config)

    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Train and evaluate model
    val_loss = train_and_evaluate(model, train_loader, val_loader)

    return val_loss


parser = argparse.ArgumentParser()

parser.add_argument("--data_directory", type=str, help="Location of training data")

parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for training",
)

args = parser.parse_args()

# Example default config without kernel_sizes (you can fill your defaults here)
default_config = {
    "input_channels": 1,
    "filters": 16,
    "dropout": 0.05,
    "batch_normalisation": True,
    "layers": 6,
    "output_activation": None,
    # kernel_sizes will be overridden by trial suggestions
}

num_layers = 6  # your UNet encoding layers count

ROOT_DIR = args.data_directory

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load data
data_train = np.load(os.path.join(ROOT_DIR, "data_train.npy"))
labels_train = np.load(os.path.join(ROOT_DIR, "labels_train.npy"))

data_validation = np.load(os.path.join(ROOT_DIR, "data_validation.npy"))
labels_validation = np.load(os.path.join(ROOT_DIR, "labels_validation.npy"))

print(f"Data train shape: {data_train.shape}")
print(f"Labels train shape: {labels_train.shape}")
print(f"Data validation shape: {data_validation.shape}")
print(f"Labels validation shape: {labels_validation.shape}")

# Convert numpy arrays to PyTorch tensors
data_train = torch.FloatTensor(data_train).permute(
    0, 3, 1, 2
)  # (N, H, W, C) -> (N, C, H, W)
print(f"dt: {data_train.shape}")
labels_train = torch.FloatTensor(labels_train).permute(0, 3, 1, 2)
print(f"lt: {labels_train.shape}")
data_validation = torch.FloatTensor(data_validation).permute(0, 3, 1, 2)
print(f"dv: {data_validation.shape}")
labels_validation = torch.FloatTensor(labels_validation).permute(0, 3, 1, 2)
print(f"lv: {labels_validation.shape}")

# Create datasets and dataloaders
train_dataset = TensorDataset(data_train, labels_train)
val_dataset = TensorDataset(data_validation, labels_validation)

batch_size = args.batch_size
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
best_params = study.best_params

print("Best kernel sizes:", best_params)

with open("best_params.json", "w") as f:
    json.dump(best_params, f)
