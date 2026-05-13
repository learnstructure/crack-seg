import segmentation_models_pytorch as smp
from crack_seg.config import NUM_CLASSES

def get_model():
    model = smp.Unet(
        # EfficientNet is more "edge-aware" than ResNet
        encoder_name="efficientnet-b4", 
        encoder_weights="imagenet",
        # Enable attention to mimic the "Edge-Body" focus
        decoder_attention_type="scse", 
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None, 
    )
    return model