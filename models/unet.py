import segmentation_models_pytorch as smp
from config import ENCODER_NAME, PRETRAINED, NUM_CLASSES


def get_model():
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights="imagenet" if PRETRAINED else None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,  # We'll apply sigmoid in loss
    )
    return model
