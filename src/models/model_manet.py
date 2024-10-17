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

        # for i, _ in enumerate(decoder_channels):
        #     decoder_channels[i] = decoder_channels[i]**(encoder_depth-i)
        
        decoder_channels = [x*(2*(encoder_depth-i)) for i,x  in enumerate(decoder_channels)]
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
