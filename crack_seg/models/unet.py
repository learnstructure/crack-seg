import segmentation_models_pytorch as smp
from crack_seg.config import NUM_CLASSES


def get_model():
    """Create a U-Net model with a pretrained ResNet101 encoder."""
    model = smp.Unet(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    return model
