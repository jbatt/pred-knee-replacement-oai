# Import libraries

import os
import sys
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy as np
from datetime import datetime

import wandb  # Import Weights and Biases for tracking model training

# Include src directory in path to import custom modules
if '..\\src' not in sys.path:
    sys.path.append('..\\src')

from models.model_unet import UNet3DMulticlass
from utils.utils import read_hyperparams
from data.datasets import KneeSegDataset3DMulticlass
from models.evaluation import bce_dice_loss #, dice_coefficient, batch_dice_coeff
from models.train import train_loop, validation_loop 


# Define data directory
DATA_DIRECTORY = '../data/oai_subset'
DATA_TRAIN_DIRECTORY = '../data/oai_subset/train'
DATA_VALID_DIRECTORY = '../data/oai_subset'

DATA_RAW_DIRECTORY = '../data/raw'
DATA_PROCESSED_DIRECTORY = '../data/processed'
DATA_INTERIM_DIRECTORY = '../data/processed'

RESULTS_PATH = '../results'
MODELS_PATH = '../models'
MODELS_CHECKPOINTS_PATH = '../models/checkpoints'


# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read in hyperparams
hyperparams = read_hyperparams('..\\src\\models\\hyperparams_unet.txt')
print(hyperparams)


# Get paths for training and and validation data
# Return file name from filepath
train_paths = np.array([os.path.basename(i).split('.')[0] for i in glob.glob(f'{DATA_TRAIN_DIRECTORY}/*.im')])
val_paths = np.array([os.path.basename(i).split('.')[0] for i in glob.glob(f'{DATA_VALID_DIRECTORY}/*.im')])

# Set transforms
if hyperparams['transforms'] == "True":
    # Let's try a horizontal flip transform
    transform = transforms.functional.hflip
else:
    transform = None


# Define PyTorch datasets and dataloader

# Define datasets
train_dataset = KneeSegDataset3DMulticlass(train_paths, DATA_DIRECTORY, transform=transform)
validation_dataset = KneeSegDataset3DMulticlass(val_paths, DATA_DIRECTORY, split='valid')

# Define dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=int(hyperparams['batch_size']), num_workers = 1, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=2, num_workers = 1, shuffle=False)


# Create model - 4 output channels for 4 classes
model = UNet3DMultiClass(1, 4, 16)

# Load model to device
model.to(device)

# Specifiy criterion and optimiser
loss_fn = bce_dice_loss
l_rate = hyperparams['l_rate']
optimizer = optim.Adam(model.parameters(), lr=l_rate)

# How long to train for?
num_epochs = int(hyperparams['num_epochs'])

# Threshold for predicted segmentation mask
pred_threshold = hyperparams['threshold']

# start a new wandb run to track this script - LOG IN ON CONSOLE BEFORE RUNNING
wandb.init(
    # set the wandb project where this run will be logged
    project="oai_subset_knee_seg_unet",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": l_rate,
    "architecture": "3D UNet",
    "kernel_num": 16,
    "dataset": "IWOAI",
    "epochs": num_epochs,
    "threshold": pred_threshold,
    }
)



# Use multiple gpu in parallel if available
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)


# Define model training fucntion using previously defined training and validation loops

# Capture training start time for output data files
train_start = str(datetime.now())
train_start_file = train_start.replace(" ", "-").replace(".","").replace(":","_")

# Initialise minimum validation loss as infinity
min_validation_loss = float('inf')

# Model training
print(f"TRAINING MODEL \n-------------------------------")

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")

    train_loss, avg_train_dice = train_loop(train_dataloader, device, model, loss_fn, optimizer, pred_threshold)
    validation_loss, avg_validation_dice = validation_loop(validation_dataloader, device, model, loss_fn, pred_threshold)

    # log to wandb
    wandb.log({"Train Loss": train_loss, "Train Dice Score": avg_train_dice,
                  "Val Loss": validation_loss, "Val Dice Score": avg_validation_dice})
    
    # save as best if val loss is lowest so far
    if validation_loss < min_validation_loss:
        print(f'Validation Loss Decreased({min_validation_loss:.6f}--->{validation_loss:.6f}) \t Saving The Model')
        model_path = f"{MODELS_CHECKPOINTS_PATH}/{hyperparams['run_name']}_best_E.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Best epoch yet: {epoch}")
        
        # reset min as current
        min_validation_loss = validation_loss


# Once training is done, save final model
model_path = f"{MODELS_CHECKPOINTS_PATH}/{hyperparams['run_name']}.pth"
torch.save(model.state_dict(), model_path)

wandb.finish()

print("Done!")