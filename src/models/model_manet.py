import torch.nn as nn
import segmentation_models_pytorch_3d as smp


class MANet3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels, encoder, encoder_depth):
        super(MANet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.encoder = encoder
        self.encoder_depth = encoder_depth

        decoder_channels = [num_kernels] * encoder_depth

        # Could use this later to automatically update the decoder channels for values of 
        # num_kernels and encoder depth - i.e. doubles every layer
        decoder_channels = [x*(2**(encoder_depth-(i+1))) for i,x  in enumerate(decoder_channels)]
        print(decoder_channels)


        self.model = smp.MAnet(
            encoder_name=self.encoder,
            encoder_depth=self.encoder_depth,
            encoder_weights=None,
            decoder_channels=(128,64,32,16),
            in_channels=1,
            classes=out_channels
        ) 

    def forward(self, x):
        output = self.model(x)

        return output 
