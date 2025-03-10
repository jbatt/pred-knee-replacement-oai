import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from models.unet import UNet3D
from models.vnet import VNet
from dataset import KneeSegDataset3DMulticlass



# TODO: tidy up create model function
# TODO: save data as 3D numpy array in folder set by model name and date



def create_model(model_name):
    if 'unet' in model_name.lower():
        model = UNet3D(in_channels=1, out_channels=3)  # Adjust channels as needed
    elif 'vnet' in model_name.lower():
        model = VNet(in_channels=1, out_channels=3)  # Adjust channels as needed
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    return model





def main(args):
    # Create output directory
    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and load model
    model = create_model(args.model_name)
    model.load_state_dict(torch.load(args.model_weights))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dataset and dataloader
    test_dataset = KneeSegDataset3DMulticlass(
        data_dir=args.test_dir,
        test=True
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Process each image
    with torch.no_grad():
        for batch in test_loader:
            image = batch['image'].to(device)
            file_path = Path(batch['file_path'][0])
            affine = batch['affine']
            
            # Forward pass
            prediction = model(image)
            
            # Add your post-processing and saving logic here
