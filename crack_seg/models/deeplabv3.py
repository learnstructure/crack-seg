import segmentation_models_pytorch as smp
from crack_seg.config import NUM_CLASSES


def get_model():
    model = smp.DeepLabV3(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    return model
