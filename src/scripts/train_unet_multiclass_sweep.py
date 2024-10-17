# Import libraries

import os
import sys
import glob

import torch
import torch.nn as nn
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
from utils.utils import read_hyperparams, EarlyStopper
from data.datasets import KneeSegDataset3DMulticlass
from src.metrics.evaluation import ce_dice_loss_multi_batch #, dice_coefficient, batch_dice_coeff
from src.trainer.trainer import train_loop, validation_loop 


# Set running environment (True for HPC, False for local)
HPC_FLAG = sys.argv[1]
HPC_FLAG

NUM_CLASSES = 5

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")


###############################################################################################
# SET UP TRAIN AND VALIDATION PATHS TO IMAGES AND MASKS
###############################################################################################

def define_dataset_paths(hpc): 

    if hpc == "1":
        # Define data directory - ARC4
        # Using absolute paths because ray-tune changing working directory
        data_dir = '/nobackup/scjb/data/oai_subset'
        data_train_dir = '/nobackup/scjb/data/oai_subset/train'
        data_valid_dir = '/nobackup/scjb/data/oai_subset/valid'
        results_dir = '/home/home02/scjb/pred-knee-replacement-oai/results'
        models_dir = '/home/home02/scjb/pred-knee-replacement-oai/models'
        models_checkpoints_dir = '/home/home02/scjb/pred-knee-replacement-oai/models/checkpoints'

    else:
        # Define data directory for local runs
        data_dir = 'C:\\Users\\james\\OneDrive - University of Leeds\\1. Projects\\1.1 PhD\\1.1.1 Project\\Data\\OAI Subset\\'
        data_train_dir = 'C:\\Users\\james\\OneDrive - University of Leeds\\1. Projects\\1.1 PhD\\1.1.1 Project\\Data\\OAI Subset\\train'
        data_valid_dir = 'C:\\Users\\james\\OneDrive - University of Leeds\\1. Projects\\1.1 PhD\\1.1.1 Project\\Data\\OAI Subset\\valid'
        results_dir = '../results'
        models_dir = '../models'
        models_checkpoints_dir = '../models/checkpoints'

    # Get paths for training and and validation data
    # Return file name from filepath
    train_paths = np.array([os.path.basename(i).split('.')[0] for i in glob.glob(f'{data_train_dir}/*.im')])
    val_paths = np.array([os.path.basename(i).split('.')[0] for i in glob.glob(f'{data_valid_dir}/*.im')])


    return data_dir, models_checkpoints_dir, train_paths, val_paths  




#####################################################################################
# SET HYPERPARAMETERS
#####################################################################################

# WandB hyperparams settings
sweep_configuration = {
    "method": "bayes",
    "name": "sweep_40_epoch",
    "metric": {"goal": "minimize", "name": "Val Loss"},
    "parameters": {
        "lr": {"max": 0.005, "min": 0.0005},
        "batch_size": {"values": [1]},
        "weight_decay": {"values": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]},
        "num_epochs": {"values": [40]},
        "transforms": {"values": [True]}
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="oai_subset_knee_seg_unet-sweep")


      


#####################################################################################
# START TRAINING RUN
#####################################################################################

def main():

    run = wandb.init()

    # Capture training start time for output data files
    train_start = str(datetime.now())
    train_start_file = train_start.replace(" ", "-").replace(".","").replace(":","_")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")    

    print(f"WandB run name: {run.name}")
    print(f"hyperparams = {run.config}")       

    lr = wandb.config.lr
    batch_size = wandb.config.batch_size
    weight_decay = wandb.config.weight_decay
    num_epochs = wandb.config.num_epochs 
    
    # Set transforms
    if wandb.config.transforms == True:
        
        # Horizontal flip transform
        transform = transforms.functional.hflip
    else:
        transform = None
    
    print(f"Current hyperparameter values:\n {wandb.config}")

    # Define train, validation and output direcotry path
    data_dir, models_checkpoints_dir, train_paths, val_paths = define_dataset_paths(hpc=HPC_FLAG)

    # Define PyTorch datasets and dataloader

    # Define datasets
    train_dataset = KneeSegDataset3DMulticlass(train_paths, data_dir, num_classes=NUM_CLASSES, transform=transform)
    validation_dataset = KneeSegDataset3DMulticlass(val_paths, data_dir, num_classes=NUM_CLASSES, split='valid')

    # Define dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=int(batch_size), num_workers = 1, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=2, num_workers = 1, shuffle=False)


    # Create model - 5 output channels for 5 classes
    model = UNet3DMulticlass(1, NUM_CLASSES, 16)

    # Load model to device
    print(f"Loading model to device: {device}")
    model.to(device)

    # Specifiy criterion and optimiser
    loss_fn = ce_dice_loss_multi_batch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True,
                                                              threshold=0.001, threshold_mode='abs')
    
    early_stopper = EarlyStopper(patience=5, min_delta=0.001)


    # Use multiple gpu in parallel if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)


    # Initialise minimum validation loss as infinity
    min_validation_loss = float('inf')

    # Model training
    print(f"TRAINING MODEL \n-------------------------------")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        train_loss, avg_train_dice, avg_train_dice_all = train_loop(train_dataloader, device, model, loss_fn, optimizer, num_classes=NUM_CLASSES)
        validation_loss, avg_validation_dice, avg_validation_dice_all, lr_scheduler = validation_loop(validation_dataloader, device, model, loss_fn, lr_scheduler, num_classes=NUM_CLASSES)

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
            model_path = os.path.join(models_checkpoints_dir, f"{train_start_file}_multiclass_{run.name}_best_E.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Best epoch yet: {epoch + 1}")
            
            # reset min as current
            min_validation_loss = validation_loss

        # Save model if early stopping triggered
        if early_stopper.early_stop(validation_loss):   
            print(f'Early stopping triggered! ({min_validation_loss:.6f}--->{validation_loss:.6f}) \t Saving The Model')
            model_path = os.path.join(models_checkpoints_dir, f"{train_start_file}_multiclass_{run.name}_early_stop_E{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Early stop epoch: {epoch + 1}") 



    # Once training is done, save final model
    model_path = os.path.join(models_checkpoints_dir, f"{train_start_file}_{run.name}_final.pth")
    torch.save(model.state_dict(), model_path)



wandb.agent(sweep_id, function=main, count=8)
