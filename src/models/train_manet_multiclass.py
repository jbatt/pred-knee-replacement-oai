# Import libraries

import os
import sys
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import segmentation_models_pytorch_3d as smp

import numpy as np
from datetime import datetime

import wandb  # Import Weights and Biases for tracking model training

# Include src directory in path to import custom modules
if '../src' not in sys.path:
    sys.path.append('../src')

print(sys.path)

from models.model_unet import UNet3DMulticlass
from utils.utils import read_hyperparams, EarlyStopper
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
hyperparams = read_hyperparams('../src/models/hyperparams_manet.txt')
print(hyperparams)


# Get paths for training and and validation data
# Return file name from filepath
train_paths = np.array([os.path.basename(i).split('.')[0] for i in glob.glob(f'{DATA_TRAIN_DIRECTORY}/*.im')])
val_paths = np.array([os.path.basename(i).split('.')[0] for i in glob.glob(f'{DATA_VALID_DIRECTORY}/*.im')])

# Set transforms
if hyperparams['transforms'] == "True":
    # Horizontal flip transform
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
model  = smp.MAnet(
    encoder_name='resnet34', # choose encoder, e.g. resnet34
    encoder_depth = 4, # matches unet architecture
    encoder_weights = None,
    decoder_channels = (128, 64, 32, 16), # matches unet architecture
    in_channels=1,                  # model input channels (1 for gray-scale volumes, 3 for RGB, etc.)
    classes=NUM_CLASSES,                      # model output channels (number of classes in your dataset)
)


# Load model to device
print(f"Loading model to device: {device}")
model.to(device)

# Specifiy criterion and optimiser
loss_fn = ce_dice_loss_multi_batch
l_rate = hyperparams['l_rate']
optimizer = optim.Adam(model.parameters(), lr=l_rate)

# Removed for now
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True,
#                                                               threshold=0.001, threshold_mode='abs')

early_stopper = EarlyStopper(patience=4, min_delta=0.001)

# How long to train for?
num_epochs = int(hyperparams['num_epochs'])

# Threshold for predicted segmentation mask
pred_threshold = hyperparams['threshold']

# start a new wandb run to track this script - LOG IN ON CONSOLE BEFORE RUNNING
wandb.init(
    # set the wandb project where this run will be logged
    project="oai_subset_knee_seg_MAnet",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": l_rate,
    "architecture": "3D MAnet No Pre-Train",
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

    train_loss, avg_train_dice, avg_train_dice_all = train_loop(train_dataloader, device, model, loss_fn, optimizer, num_classes=NUM_CLASSES)
    validation_loss, avg_validation_dice, avg_validation_dice_all = validation_loop(validation_dataloader, device, model, loss_fn, num_classes=NUM_CLASSES)

    # log to wandb
    wandb.log({
        "Train Loss": train_loss, 
        "Train Dice Score": avg_train_dice,
        "Train Dice Score (Background)": avg_train_dice_all[0],
        "Train Dice Score (Femoral Cart.)": avg_train_dice_all[1],
        "Train Dice Score (Tibial Cart.)": avg_train_dice_all[2],
        "Train Dice Score (Patellar Cart.)": avg_train_dice_all[3],
        "Train Dice Score (Meniscus)": avg_train_dice_all[4],
        "Val Loss": validation_loss, 
        "Val Dice Score": avg_validation_dice,
        "Val Dice Score (Background)": avg_validation_dice_all[0],
        "Val Dice Score (Femoral Cart.)": avg_validation_dice_all[1],
        "Val Dice Score (Tibial Cart.)": avg_validation_dice_all[2],
        "Val Dice Score (Patellar Cart.)": avg_validation_dice_all[3],
        "Val Dice Score (Meniscus)": avg_validation_dice_all[4],
    })
    
    # Save as best if val loss is lowest so far
    if validation_loss < min_validation_loss:
        print(f'Validation Loss Decreased({min_validation_loss:.6f}--->{validation_loss:.6f}) \t Saving The Model')
        model_path = os.path.join(MODELS_CHECKPOINTS_PATH, f"{train_start_file}_{hyperparams["run_name"]}_best_E{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Best epoch yet: {epoch + 1}")
        
        # reset min as current
        min_validation_loss = validation_loss

    # Save model if early stopping triggered
    if early_stopper.early_stop(validation_loss):   
        print(f'Early stopping triggered! ({min_validation_loss:.6f}--->{validation_loss:.6f}) \t Saving The Model')
        model_path = os.path.join(MODELS_CHECKPOINTS_PATH, f"{train_start_file}_{hyperparams["run_name"]}_early_stop_E{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Early stop epoch: {epoch + 1}") 



# Once training is done, save final model
model_path = os.path.join(MODELS_CHECKPOINTS_PATH, f"{train_start_file}_{hyperparams["run_name"]}_final.pth")
torch.save(model.state_dict(), model_path)

wandb.finish()

print("Done!")
