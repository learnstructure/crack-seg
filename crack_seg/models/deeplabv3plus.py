
import segmentation_models_pytorch as smp
from crack_seg.config import ENCODER_NAME, ENCODER_WEIGHTS, NUM_CLASSES

def get_model():
    """
    Returns a DeepLabV3+ model from the segmentation-models-pytorch library.
    """
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    return model
