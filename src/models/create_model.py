from models.model_manet import MANet3D
from models.model_unet import UNet3DMulticlass
from models.model_swinunetr import SwinUNETR


def create_model(
        input_model_arg, 
        in_channels=1, 
        out_channels=5, 
        num_kernels=16, 
        encoder="resnet34", 
        encoder_depth=4,
        img_size=(288,288,160),
        feature_size=48
    ):
    
    if input_model_arg == "manet":
        print(f"Creating manet model...")
        model = MANet3D(in_channels, out_channels, num_kernels, encoder, encoder_depth)
        print(f"Model created:\n{model}")
    
    elif input_model_arg == "unet":
        print(f"Creating unet model...")
        model = UNet3DMulticlass(in_channels, out_channels, num_kernels)
        print(f"Model created:\n{model}")
    
    elif input_model_arg == "swin_unetr":
        print(f"Creating swin_unetr model...")
        model = SwinUNETR(img_size=img_size, 
                          in_channels=in_channels, 
                          out_channels=out_channels, 
                          feature_size=feature_size,
                          use_checkpoint=True)
        print(f"Model created:\n{model}")

    return model