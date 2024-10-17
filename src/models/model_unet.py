import torch
import torch.nn as nn

# Unet Conv block
# ConvBlock inherits from the nn.Module class
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # Padding of 1 used so input height/width = output height/width
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        # Bath normalisation immproves training efficiency
        self.bn = nn.BatchNorm3d(out_channels)

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

    # Apply the convolution, batch normalisation and ReLU activation function
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x


# Encoder block, using two conv blocks    
class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncBlock, self).__init__()

        # mid_channels = int(out_channels/2)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x
    
# Unet Upsampling block
class UpConvBlock(nn.Module):   
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()

        # Transpose convolutions
        # ConvTransose3D is an alternative to upsampling
        self.upconv = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3,
                                         stride=2, padding=1, output_padding=1)

        post_skip = int(in_channels*1.5)
        self.conv1 = ConvBlock(post_skip, out_channels) # after adding skip connection
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat([x, skip_connection], dim=1)  # Concatenate upsampled with skip connection
        x = self.conv1(x)
        x = self.conv2(x)

        return x
    
# Define 3D U-Net architecture
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels_1):
        super(UNet3D, self).__init__()

        # Encoder
        # Encoder consists of three encoding blocks which double the number of kernels with each block
        self.enc1 = EncBlock(in_channels, num_kernels_1)
        self.enc2 = EncBlock(num_kernels_1, num_kernels_1*2)
        self.enc3 = EncBlock(num_kernels_1*2, num_kernels_1*4)

        # Pooling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = EncBlock(num_kernels_1*4, num_kernels_1*8)

        # Decoder (upsampling)
        self.dec3 = UpConvBlock(num_kernels_1*8, num_kernels_1*4)
        self.dec2 = UpConvBlock(num_kernels_1*4, num_kernels_1*2)
        self.dec1 = UpConvBlock(num_kernels_1*2, num_kernels_1)

        # Output
        self.out_conv = nn.Conv3d(num_kernels_1, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pool = self.pool(enc1)
        enc2 = self.enc2(pool)
        pool = self.pool(enc2)
        enc3 = self.enc3(pool)
        pool = self.pool(enc3)

        # Bottleneck
        x = self.bottleneck(pool)

        # Decoder
        x = self.dec3(x, enc3)
        x = self.dec2(x, enc2)
        x = self.dec1(x, enc1)

        # Output
        x = self.out_conv(x)
        x = self.sigmoid(x)

        return x
    


# Define Multiclass 3D U-Net architecture
class UNet3DMulticlass(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels_1):
        super(UNet3DMulticlass, self).__init__()

        # Encoder
        # Encoder consists of three encoding blocks which double the number of kernels with each block
        self.enc1 = EncBlock(in_channels, num_kernels_1)
        self.enc2 = EncBlock(num_kernels_1, num_kernels_1*2)
        self.enc3 = EncBlock(num_kernels_1*2, num_kernels_1*4)

        # Pooling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = EncBlock(num_kernels_1*4, num_kernels_1*8)

        # Decoder (upsampling)
        self.dec3 = UpConvBlock(num_kernels_1*8, num_kernels_1*4)
        self.dec2 = UpConvBlock(num_kernels_1*4, num_kernels_1*2)
        self.dec1 = UpConvBlock(num_kernels_1*2, num_kernels_1)

        # Output
        self.out_conv = nn.Conv3d(num_kernels_1, out_channels, kernel_size=1)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pool = self.pool(enc1)
        enc2 = self.enc2(pool)
        pool = self.pool(enc2)
        enc3 = self.enc3(pool)
        pool = self.pool(enc3)

        # Bottleneck
        x = self.bottleneck(pool)

        # Decoder
        x = self.dec3(x, enc3)
        x = self.dec2(x, enc2)
        x = self.dec1(x, enc1)

        # Output
        x = self.out_conv(x)
        # x = self.softmax(x)

        return x