
import segmentation_models_pytorch as smp
from crack_seg.config import ENCODER_NAME, NUM_CLASSES, PRETRAINED

def get_model():
    """
    Returns a U-Net++ model from the segmentation-models-pytorch library.
    """
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER_NAME,
        encoder_weights="imagenet" if PRETRAINED else None,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    return model
