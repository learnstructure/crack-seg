import segmentation_models_pytorch as smp
from crack_seg.config import NUM_CLASSES


def get_model():
    """
    Returns a U-Net++ model from the segmentation-models-pytorch library.
    """
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        decoder_attention_type="scse",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    return model
