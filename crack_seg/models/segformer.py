import segmentation_models_pytorch as smp
from crack_seg.config import NUM_CLASSES


def get_model():
    """
    Returns a SegFormer model for crack segmentation.
    Encoder: mit_b2 (Mix Transformer) pre‑trained on ImageNet.
    """
    model = smp.Segformer(
        encoder_name="mit_b2",
        encoder_weights="imagenet",  # pre‑trained on ImageNet
        in_channels=3,
        classes=NUM_CLASSES,  # 1 for binary segmentation
        activation=None,  # output raw logits
    )
    return model
