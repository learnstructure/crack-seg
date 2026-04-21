import torch
import torch.nn as nn


class SegNet(nn.Module):
    """
    Faster SegNet implementation using bilinear upsampling + convolution
    instead of MaxUnpool2d.
    """

    def __init__(self, in_channels=3, out_channels=1):
        super(SegNet, self).__init__()

        # ---------- Encoder ----------
        # Block 1
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------- Decoder (using bilinear upsampling + conv) ----------
        # Upsample 1: from 256 to 256 (scale factor 2)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Upsample 2: from 128 to 128 (scale factor 2)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Upsample 3: from 64 to 64 (scale factor 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc_conv1(x)
        x1_pooled = self.pool1(x1)

        x2 = self.enc_conv2(x1_pooled)
        x2_pooled = self.pool2(x2)

        x3 = self.enc_conv3(x2_pooled)
        x3_pooled = self.pool3(x3)

        # Decoder (bilinear upsampling)
        x3_up = self.up3(x3_pooled)
        x3 = self.dec_conv3(x3_up)

        x2_up = self.up2(x3)
        x2 = self.dec_conv2(x2_up)

        x1_up = self.up1(x2)
        x1 = self.dec_conv1(x1_up)

        return x1


def get_model(in_channels=3, out_channels=1):
    """Returns a SegNet model instance."""
    return SegNet(in_channels=in_channels, out_channels=out_channels)
