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
if '../src' not in sys.path:
    sys.path.append('../src')

print(sys.path)

from models.model_unet import UNet3DMulticlass
from utils.utils import read_hyperparams
from data.datasets import KneeSegDataset3DMulticlass
from models.evaluation import ce_dice_loss_multi_batch #, dice_coefficient, batch_dice_coeff
from models.train import train_loop, validation_loop 


# Set running environment (True for HPC, False for local)
HPC_FLAG = sys.argv[1]
HPC_FLAG

if HPC_FLAG == "1":
    # Define data directory - ARC4
    # Using absolute paths because ray-tune changing working directory
    DATA_DIRECTORY = '/nobackup/scjb/data/oai_subset'
    DATA_TRAIN_DIRECTORY = '/nobackup/scjb/data/oai_subset/train'
    DATA_VALID_DIRECTORY = '/nobackup/scjb/data/oai_subset/valid'
    RESULTS_PATH = '/home/home02/scjb/pred-knee-replacement-oai/results'
    MODELS_PATH = '/home/home02/scjb/pred-knee-replacement-oai/models'
    MODELS_CHECKPOINTS_PATH = '/home/home02/scjb/pred-knee-replacement-oai/models/checkpoints'

else:
    # Define data directory for local runs
    DATA_DIRECTORY = 'C:\\Users\\james\\OneDrive - University of Leeds\\1. Projects\\1.1 PhD\\1.1.1 Project\\Data\\OAI Subset\\'
    DATA_TRAIN_DIRECTORY = 'C:\\Users\\james\\OneDrive - University of Leeds\\1. Projects\\1.1 PhD\\1.1.1 Project\\Data\\OAI Subset\\train'
    DATA_VALID_DIRECTORY = 'C:\\Users\\james\\OneDrive - University of Leeds\\1. Projects\\1.1 PhD\\1.1.1 Project\\Data\\OAI Subset\\valid'

    RESULTS_PATH = '../results'
    MODELS_PATH = '../models'
    MODELS_CHECKPOINTS_PATH = '../models/checkpoints'

 

NUM_CLASSES = 5

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

# Read in hyperparams
hyperparams = read_hyperparams('../src/models/hyperparams_unet.txt')
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
train_dataset = KneeSegDataset3DMulticlass(train_paths, DATA_DIRECTORY, num_classes=NUM_CLASSES, transform=transform)
validation_dataset = KneeSegDataset3DMulticlass(val_paths, DATA_DIRECTORY, num_classes=NUM_CLASSES, split='valid')

# Define dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=int(hyperparams['batch_size']), num_workers = 1, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=2, num_workers = 1, shuffle=False)


# Create model - 5 output channels for 5 classes
model = UNet3DMulticlass(1, 5, 16)

# Load model to device
print(f"Loading model to device: {device}")
model.to(device)

# Specifiy criterion and optimiser
loss_fn = ce_dice_loss_multi_batch
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

    train_loss, avg_train_dice = train_loop(train_dataloader, device, model, loss_fn, optimizer, pred_threshold, num_classes=NUM_CLASSES)
    validation_loss, avg_validation_dice = validation_loop(validation_dataloader, device, model, loss_fn, pred_threshold, num_classes=NUM_CLASSES)

    # log to wandb
    wandb.log({"Train Loss": train_loss, "Train Dice Score": avg_train_dice,
                  "Val Loss": validation_loss, "Val Dice Score": avg_validation_dice})
    
    # save as best if val loss is lowest so far
    if validation_loss < min_validation_loss:
        print(f'Validation Loss Decreased({min_validation_loss:.6f}--->{validation_loss:.6f}) \t Saving The Model')
        model_path = os.path.join(MODELS_CHECKPOINTS_PATH, f"multiclass_{hyperparams['run_name']}_best_E.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Best epoch yet: {epoch}")
        
        # reset min as current
        min_validation_loss = validation_loss


# Once training is done, save final model
model_path = os.path.join(MODELS_CHECKPOINTS_PATH, f"{hyperparams['run_name']}_final.pth")
torch.save(model.state_dict(), model_path)

wandb.finish()

print("Done!")
