import argparse
import os
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from torchvision import transforms as tfs
import random

from unet.models import CustomCallback, UNet


class CustomImageDataset(Dataset):
    def __init__(self, imgs, transform=None, target_transform=None):
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        if self.transform:
            image, label = add_contaminants.forward(image)
            image, label = pre_process_image.forward(image, label)
        return image, label


class PreprocessImage(torch.nn.Module):
    def forward(self, img=None, label=None, normalise_max=300):

        if img is None:
            assert False
        if label is None:
            assert False

        data_max = torch.max(img)
        normalisation_factor = normalise_max / data_max
        img_normalised = img * normalisation_factor
        label_normalised = label * normalisation_factor

        return torch.log1p(img_normalised), torch.log1p(label_normalised)


class AddContaminants(torch.nn.Module):
    def forward(self, img=None, label=None, Jmag_range=(1, 16)):

        if img is None:
            assert False

        # # Add the random order 0s
        # print("within addContaminants")
        # print("img shape:", img.shape)
        # print("order0 shape:", order_zeros.shape)

        num_spectra, target_rows, target_cols = img.shape
        num_order0s, array_rows, array_cols = order_zeros.shape
        contam_list = []
        N_contaminants = random.randint(0, 10)

        new_img = img
        new_label = torch.zeros(img.shape)

        for _ in range(N_contaminants):

            # Randomize the top-left position of the array of ones
            start_row = random.randint(-array_rows + 1, target_rows)
            start_col = random.randint(700, target_cols)  # POM starts at col 700

            # Randomly choose a multiplication factor from the specified range
            factor = random.uniform(*Jmag_range)

            order_ = random.randint(0, num_order0s - 1)

            contaminant = (
                order_zeros[order_, :, :] * 1e4
            )  # Scaling factor based on observations

            # Determine overlap region in the target array
            end_row = start_row + array_rows
            end_col = start_col + array_cols

            # Clip indices to be within the bounds of the target array
            target_start_row = max(start_row, 0)
            target_start_col = max(start_col, 0)
            target_end_row = min(end_row, target_rows)
            target_end_col = min(end_col, target_cols)

            # Determine the corresponding region of the array of ones
            array_start_row = max(0, -start_row)
            array_start_col = max(0, -start_col)
            array_end_row = array_start_row + (target_end_row - target_start_row)
            array_end_col = array_start_col + (target_end_col - target_start_col)

            new_img[
                :, target_start_row:target_end_row, target_start_col:target_end_col
            ] += (
                factor
                * contaminant[
                    array_start_row:array_end_row, array_start_col:array_end_col
                ]
            )

            new_label[
                :, target_start_row:target_end_row, target_start_col:target_end_col
            ] += (
                factor
                * contaminant[
                    array_start_row:array_end_row, array_start_col:array_end_col
                ]
            )
            # print("end of loop")
        return new_img, new_label


def train_model(
    model, train_loader, val_loader, num_epochs=200, learning_rate=0.001, patience=10
):
    """Train the UNet model."""

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, min_lr=0.00001
    )

    callback = CustomCallback(os.path.join(ROOT_DIR, SAVE_DIR))

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(
                    device
                )
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Custom callback
        callback.on_epoch_end(epoch, avg_train_loss, avg_val_loss)

        # Early stopping and model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save(
                model.state_dict(),
                os.path.join(ROOT_DIR, SAVE_DIR, "trained_model.pth"),
            )
            print(
                f"Epoch {epoch+1}: New best model saved with val_loss: {avg_val_loss:.6f}"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {avg_train_loss:.6f}, "
                f"Val Loss: {avg_val_loss:.6f}"
            )

    return {"loss": train_losses, "val_loss": val_losses}


parser = argparse.ArgumentParser()

parser.add_argument("--data_directory", type=str, help="Location of training data")

parser.add_argument(
    "--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Number of epochs to train",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for training",
)

args = parser.parse_args()

ROOT_DIR = args.data_directory
SAVE_DIR = (
    f"save_{args.learning_rate}_{args.epochs}_{args.batch_size}_"
    f"{datetime.now().replace(microsecond=0).isoformat()}"
)
os.makedirs(os.path.join(ROOT_DIR, SAVE_DIR), exist_ok=True)

print(f"Root directory: {ROOT_DIR}")
print(f"Save directory: {SAVE_DIR}")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

add_contaminants = AddContaminants()
pre_process_image = PreprocessImage()

# Load data
data_train = np.load(os.path.join(ROOT_DIR, "clean_train.npy"))
data_train = data_train[:800, :, :]
data_validation = np.load(os.path.join(ROOT_DIR, "clean_validation.npy"))
data_validation = data_validation[:100, :, :]

# Load order zeros
order_zeros = np.load(os.path.join(ROOT_DIR, "contaminants.npy"))

print(f"Data train shape: {data_train.shape}")
print(f"Data validation shape: {data_validation.shape}")
print(f"Order zero shape: {order_zeros.shape}")

data_train = torch.FloatTensor(data_train).unsqueeze(-1)
data_validation = torch.FloatTensor(data_validation).unsqueeze(-1)

# Convert numpy arrays to PyTorch tensors
data_train = torch.FloatTensor(data_train).permute(
    0, 3, 1, 2
)  # (N, H, W, C) -> (N, C, H, W)
data_validation = torch.FloatTensor(data_validation).permute(0, 3, 1, 2)

transforms = tfs.Compose([tfs.ToTensor()])
# Create datasets and dataloaders
train_dataset = CustomImageDataset(data_train, transform=transforms)
val_dataset = CustomImageDataset(data_validation, transform=transforms)

batch_size = args.batch_size
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

# Create UNet model
unet_model = UNet(
    input_channels=1,
    filters=16,
    dropout=0.05,
    batch_normalisation=True,
    layers=6,
    output_activation=None,
    kernel_sizes=[(3, 5), (7, 5), (5, 3), (3, 5), (5, 15), (13, 3)],
)

print(
    f"Model parameters: {sum(p.numel() for p in unet_model.parameters() if p.requires_grad)}"
)

# Train the model
results = train_model(
    unet_model,
    train_loader,
    val_loader,
    num_epochs=args.epochs,
    learning_rate=args.learning_rate,
)

# Plot final results
best_model_epoch = np.argmin(results["val_loss"])
best_model_val_loss = np.min(results["val_loss"])

fig, ax = plt.subplots(1, figsize=(5, 4))

ax.plot(results["loss"], label="training loss", color="#57d4c1", marker="None")
ax.plot(results["val_loss"], label="validation loss", color="#8d57d4", marker="None")
ax.plot(
    best_model_epoch,
    best_model_val_loss,
    marker="x",
    label="best model",
    color="#d457bf",
    linestyle="None",
    markersize=5,
)

ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend()

plt.savefig(os.path.join(ROOT_DIR, SAVE_DIR, "loss_plot_varying.png"))

ax.set_yscale("log")
plt.savefig(os.path.join(ROOT_DIR, SAVE_DIR, "loss_plot_varying_log.png"))
plt.close()

print("PyTorch UNet model conversion completed!")
