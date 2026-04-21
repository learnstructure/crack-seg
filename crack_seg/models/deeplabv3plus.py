import segmentation_models_pytorch as smp
from crack_seg.config import NUM_CLASSES


def get_model():
    """
    Returns a DeepLabV3+ model from the segmentation-models-pytorch library.
    """
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    return model
