from models.model_manet import MANet3D
from models.model_unet import UNet3DMulticlass


def create_model(input_model_arg, in_channels=1, out_channels=5, num_kernels=16, encoder="resnet34", encoder_depth=4):
    
    if input_model_arg == "manet":
        model = MANet3D(in_channels, out_channels, num_kernels, encoder, encoder_depth)
    elif input_model_arg == "unet":
        model = UNet3DMulticlass(in_channels, out_channels, num_kernels)

    return model