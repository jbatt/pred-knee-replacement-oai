import torch
import torch.nn as nn
import torch.nn.functional as F

# Spatial attention module for 3D
class SAM3D(nn.Module):
    def __init__(self, bias=False):
        super(SAM3D, self).__init__()
        self.bias = bias
        self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=self.bias)

    def forward(self, x):
        # x: [B, C, D, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)

        concat = torch.cat((max_out, avg_out), dim=1)  # [B, 2, D, H, W]
        attn = torch.sigmoid(self.conv(concat))        # [B, 1, D, H, W]
        return attn * x




# Channel attention module for 3D
class CAM3D(nn.Module):
    def __init__(self, channels, r):
        super(CAM3D, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(self.channels, self.channels // self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels // self.r, self.channels, bias=True)
        )

    def forward(self, x):
        # x: [B, C, D, H, W]
        max_pool = F.adaptive_max_pool3d(x, 1).view(x.size(0), x.size(1))  # [B, C]
        avg_pool = F.adaptive_avg_pool3d(x, 1).view(x.size(0), x.size(1))  # [B, C]
        max_attn = self.linear(max_pool).view(x.size(0), x.size(1), 1, 1, 1)
        avg_attn = self.linear(avg_pool).view(x.size(0), x.size(1), 1, 1, 1)
        attn = torch.sigmoid(max_attn + avg_attn)
        return attn * x

class CBAM3D(nn.Module):
    def __init__(self, channels, r):
        super(CBAM3D, self).__init__()
        self.cam = CAM3D(channels, r)
        self.sam = SAM3D(bias=False)

    def forward(self, x):
        out = self.cam(x)
        out = self.sam(out)
        return out + x
