import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class ConvBlock(nn.Module):
    """Convolutional block for UNet."""

    def __init__(
        self, in_channels, out_channels, kernel_size, use_batch_norm=True, dropout=0.05
    ):
        super(ConvBlock, self).__init__()
        self.use_batch_norm = use_batch_norm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        if self.use_batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)

        # He normal initialization
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        return x


class EncodingBlock(nn.Module):
    """Encoding block for UNet."""

    def __init__(
        self, in_channels, out_channels, kernel_size, use_batch_norm=True, dropout=0.05
    ):
        super(EncodingBlock, self).__init__()
        self.conv_block = ConvBlock(
            in_channels, out_channels, kernel_size, use_batch_norm, dropout
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        conv_out = self.conv_block(x)
        pooled = self.pool(conv_out)
        pooled = self.dropout(pooled)
        return conv_out, pooled


class DecodingBlock(nn.Module):
    """Decoding block for UNet."""

    def __init__(
        self,
        in_channels,
        concat_channels,
        out_channels,
        kernel_size,
        use_batch_norm=True,
        dropout=0.05,
    ):
        super(DecodingBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=1
        )
        self.dropout = nn.Dropout2d(dropout)
        self.conv_block = ConvBlock(
            out_channels + concat_channels,
            out_channels,
            kernel_size,
            use_batch_norm,
            dropout,
        )

    def forward(self, x, concat_tensor):
        x = self.upconv(x)

        # Handle size mismatch between upsampled and concat tensors
        if x.shape != concat_tensor.shape:
            x = F.interpolate(
                x, size=concat_tensor.shape[2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x, concat_tensor], dim=1)
        x = self.dropout(x)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    """UNet model for image segmentation."""

    def __init__(
        self,
        input_channels=1,
        filters=16,
        dropout=0.05,
        batch_normalisation=True,
        layers=4,
        output_activation="sigmoid",
        kernel_sizes=[(3, 5), (7, 5), (5, 3), (3, 5), (5, 15), (13, 3)],
    ):
        """
        Initialize the UNet model.

        Parameters
        ----------
        input_channels : int
            Number of input channels, default is 1.
        filters : int
            The number of filters to use in the convolutional layers, default is 16.
        dropout : float
            The dropout rate, default is 0.05.
        batch_normalisation : bool
            Whether to use batch normalisation, default is True.
        layers : int
            The number of encoding and decoding layers, default is 4.
        output_activation : str
            The activation function for the output layer, either "sigmoid", "softmax", or None.
        kernel_sizes : list
            List of kernel sizes for each layer.
        """
        super(UNet, self).__init__()

        self.filters = filters
        self.dropout = dropout
        self.batch_normalisation = batch_normalisation
        self.layers = len(kernel_sizes)
        self.output_activation = output_activation
        self.kernel_sizes = kernel_sizes

        # Encoding path
        self.encoding_blocks = nn.ModuleList()
        in_channels = input_channels

        for i, kernel_size in enumerate(kernel_sizes):
            out_channels = filters * (2**i)
            self.encoding_blocks.append(
                EncodingBlock(
                    in_channels, out_channels, kernel_size, batch_normalisation, dropout
                )
            )
            in_channels = out_channels

        # Latent space
        self.latent_conv = ConvBlock(
            in_channels,
            filters * (2**self.layers),
            kernel_sizes[-1],
            batch_normalisation,
            dropout,
        )

        # Decoding path
        self.decoding_blocks = nn.ModuleList()
        in_channels = filters * (2**self.layers)

        for i in reversed(range(len(kernel_sizes))):
            out_channels = filters * (2**i)
            concat_channels = out_channels  # Skip connection channels
            kernel_size = kernel_sizes[i]

            self.decoding_blocks.append(
                DecodingBlock(
                    in_channels,
                    concat_channels,
                    out_channels,
                    kernel_size,
                    batch_normalisation,
                    dropout,
                )
            )
            in_channels = out_channels

        # Output layer
        self.output_conv = nn.Conv2d(filters, 1, kernel_size=1)

        # Output activation
        if output_activation == "sigmoid":
            self.output_activation_fn = nn.Sigmoid()
        elif output_activation == "softmax":
            self.output_activation_fn = nn.Softmax(dim=1)
        else:
            self.output_activation_fn = None

    def forward(self, x):
        # Encoding path
        skip_connections = []
        current = x

        for encoding_block in self.encoding_blocks:
            skip_conn, current = encoding_block(current)
            skip_connections.append(skip_conn)

        # Latent space
        current = self.latent_conv(current)

        # Decoding path
        for i, decoding_block in enumerate(self.decoding_blocks):
            skip_idx = len(skip_connections) - 1 - i
            current = decoding_block(current, skip_connections[skip_idx])

        # Output
        output = self.output_conv(current)

        if self.output_activation_fn:
            output = self.output_activation_fn(output)

        # Apply absolute value (matching TF version)
        output = torch.abs(output)

        return output


class CustomCallback:
    """Custom callback for tracking training progress."""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, train_loss, val_loss):
        self.losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Save losses
        logs_losses = {"loss": self.losses, "val_loss": self.val_losses}
        np.save(
            os.path.join(
                self.save_dir,
                "training_loss_varying_layers_2000_test_run_zeroth_scaled_1e7.npy",
            ),
            logs_losses,
        )

        # Plot losses
        fig, ax = plt.subplots(1, figsize=(5, 4))
        ax.plot(self.losses, label="training loss", color="#57d4c1", marker="None")
        ax.plot(
            self.val_losses, label="validation loss", color="#8d57d4", marker="None"
        )

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()

        plt.savefig(os.path.join(self.save_dir, "training_loss_plot.png"))

        ax.set_yscale("log")
        plt.savefig(os.path.join(self.save_dir, "training_loss_plot_log.png"))

        plt.close()


class UNetPredictor:
    """Class for making predictions with trained UNet model."""

    def __init__(self, model_path, model_config=None):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_config is None:
            model_config = {
                "input_channels": 1,
                "filters": 16,
                "dropout": 0.05,
                "batch_normalisation": True,
                "layers": 6,
                "output_activation": None,
                "kernel_sizes": [(9, 17), (5, 5), (5, 9), (5, 9), (5, 7), (3, 3)],
            }

        self.model = UNet(**model_config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        """
        Decode image using trained model.

        Parameters
        ----------
        image : np.ndarray
            Input image as numpy array with shape (N, H, W, C) or (H, W, C)

        Returns
        -------
        np.ndarray
            Predicted segmentation mask
        """
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = image[np.newaxis, ...]  # Add batch dimension
            image = torch.FloatTensor(image).permute(
                0, 3, 1, 2
            )  # (N, H, W, C) -> (N, C, H, W)

        image = image.to(self.device)

        with torch.no_grad():
            prediction = self.model(image)

        # Convert back to numpy and original format
        prediction = (
            prediction.permute(0, 2, 3, 1).cpu().numpy()
        )  # (N, C, H, W) -> (N, H, W, C)

        return prediction
