from models.model_manet import MANet3D
from models.model_unet import UNet3DMulticlass
from models.model_swinunetr import SwinUNETR
from models.model_segformer3d import SegFormer3D


def create_model(
        input_model_arg, 
        in_channels=1, 
        out_channels=5, 
        num_kernels=16, 
        encoder="resnet34", 
        encoder_depth=4,
        img_size=(288,288,160),
        feature_size=48,
        patch_size=None
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
        model = SwinUNETR(img_size=tuple(patch_size), 
                          in_channels=in_channels, 
                          out_channels=out_channels, 
                          feature_size=feature_size,
                          use_checkpoint=True)
        print(f"Model created:\n{model}")
    
    elif input_model_arg == "segformer3d":
        print(f"Creating segformer3d model...")
        model = SegFormer3D(in_channels = in_channels,
                            sr_ratios = [4, 2, 1, 1],
                            embed_dims = [32, 64, 160, 256],
                            patch_kernel_size = [7, 3, 3, 3],
                            patch_stride = [4, 2, 2, 2],
                            patch_padding = [3, 1, 1, 1],
                            mlp_ratios = [4, 4, 4, 4],
                            num_heads = [1, 2, 5, 8],
                            depths = [2, 2, 2, 2],
                            decoder_head_embedding_dim = 256,
                            num_classes = out_channels,
                            decoder_dropout = 0.0)
    
    print(f"Model created:\n{model}")

    return model