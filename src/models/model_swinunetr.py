import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from typing import Tuple, Union

class SwinUNETRModel(nn.Module):
    def __init__(
        self,
        img_size: Union[Tuple[int, int, int], int] = (272, 272, 160),
        in_channels: int = 1,
        out_channels: int = 5,
        feature_size: int = 48,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = False
    ):
        """
        Initialize SwinUNETR model.
        
        Args:
            img_size: Input image size
            in_channels: Number of input channels
            out_channels: Number of output channels
            feature_size: Feature size
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            dropout_path_rate: Drop path rate
            use_checkpoint: Use gradient checkpointing
        """
        super().__init__()
        
        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x)
    
    def load_pretrained_weights(self, weights_path: str):
        """
        Load pretrained weights.
        
        Args:
            weights_path: Path to pretrained weights
        """
        self.model.load_state_dict(torch.load(weights_path))