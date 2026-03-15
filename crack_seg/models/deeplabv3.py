import segmentation_models_pytorch as smp


def get_model():
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights="imagenet" if PRETRAINED else None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    return model
