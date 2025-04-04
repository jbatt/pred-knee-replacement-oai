# TODO: learning rate scheduler? 

# Import libraries
import os
import sys
import glob
import json

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Pytorch distributed training
import torch.multiprocessing as mp  # Wrap asround python's native numtiprocessing
from torch.utils.data.distributed import DistributedSampler # Distributed inpout data across GPUs
from torch.nn.parallel import DistributedDataParallel as DDP # Distribute model across GPUs
from torch.distributed import init_process_group, destroy_process_group # Initialise process group for distributed training

import numpy as np
from datetime import datetime

import wandb  # Import Weights and Biases for tracking model training
os.environ["WANDB__SERVICE_WAIT"] = "300" # Included to prevent wandb.sdk.service.service.ServiceStartTimeoutError: Timed out waiting for wandb service to start after 30.0 seconds error

import yaml # To read in the wandb config yaml file

# Include src directory in path to import custom modules
if '../src' not in sys.path:
    sys.path.append('../src')

print(sys.path)

from models.model_unet import UNet3DMulticlass
from utils.utils import EarlyStopper
from data.datasets import KneeSegDataset3DMulticlass
from metrics.loss import create_loss, ce_dice_loss_multi_batch #, dice_coefficient, batch_dice_coeff
from trainer.trainer import train_loop, validation_loop 
from models.create_model import create_model

import argparse


NUM_CLASSES = 5



###############################################################################################
# SET UP TRAIN AND VALIDATION PATHS TO IMAGES AND MASKS
###############################################################################################

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

def define_dataset_paths(hpc): 

    if hpc == "1":
        # Define data directory - Aire
        data_dir = os.path.join('/mnt', 'scratch', 'scjb', 'data', 'oai_subset')
        data_train_dir = os.path.join(data_dir, 'train')
        data_valid_dir = os.path.join(data_dir, 'valid')

        print(f"data_train_dir = {data_train_dir}")
        print(f"data_valid_dir = {data_valid_dir}")

        models_dir = os.path.join('/mnt', 'scratch', 'scjb', 'models')
        models_checkpoints_dir = os.path.join(models_dir, 'checkpoints')
        home_dir = os.path.join('/users', 'scjb')
        results_dir = os.path.join(home_dir, 'pred-knee-replacement-oai', 'results')

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

    print(f"train_paths = {train_paths}")
    print(f"val_paths = {val_paths}")

    return data_dir, models_checkpoints_dir, train_paths, val_paths  



###############################################################################################
# PARRALLELSED TRAINING SETUP - DISTRIBUTED DATA PARALLEL 
###############################################################################################
def ddp_setup(local_rank: int, world_size: int)  -> None:
    """ Set up distributed data parallel training using PyTorch's native multiprocessing library
    Args:
        rank (int): Unique identifier of each prcoess
        world_size (int): Total number of processes
    """
    # IP address of machine running rank 0 process
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialise process group - 
    # nccl backend: nvidia collective communications library - optimised for Nvidia GPUs
    print(f"\nSetting up distributed data parallel training for rank {local_rank} of {world_size}\n")
    init_process_group(backend='nccl', init_method='env://', rank=local_rank, world_size=world_size)


#####################################################################################
# START TRAINING RUN
#####################################################################################

# Distributed training function
def train(rank: int, world_size: int, config, args) -> None:
    
    # Capture training start time for output data files
    train_start = str(datetime.now())
    train_start_file = train_start.replace(" ", "-").replace(".","").replace(":","_")

    #####################################################################################
    # CREATE MODEL AND PARALLELISE IF MULTIPLE GPUS AVAILABLE
    #####################################################################################
    # Check available device
    device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device type = {device_type}")

    # Set up distributed data parallel training prcoesses 
    ddp_setup(rank, world_size)

    # Set device for each process
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    print(f"Rank {rank} device: {device}")

    # Only initialize wandb normally on the master process (rank 0)
    if rank == 0:
        print(f"Initialising wandb run for rank {rank}")
        run = wandb.init(project="oai-subset-knee-cart-seg", config=config)
    else:
        # Disable wandb logging for non-master processes
        run = wandb.init(mode="disabled")
    
    print(f"WandB run name: {run.name}")
    print(f"config = {run.config}")  

    # Set training hyperparameters from config file
    lr = wandb.config.lr
    batch_size = wandb.config.batch_size
    # weight_decay = wandb.config.weight_decay # removed weight decay for now - will include if regularisation required
    num_epochs = wandb.config.num_epochs 
    
    # Create model using wandb config hyperparams
    model = create_model(input_model_arg=args.model, 
                        in_channels =wandb.config.in_channels, 
                        out_channels=wandb.config.out_channels, 
                        num_kernels= wandb.config.num_kernels, 
                        encoder=wandb.config.encoder, # None/null used in config file if not relevant for model
                        encoder_depth=wandb.config.encoder_depth # None/null used if not relevant for model
    )

    # Use multiple gpu in parallel if available
    # if torch.cuda.device_count() > 1:
    #     device_ids = [i for i in range(torch.cuda.device_count())]
    #     model = nn.DataParallel(model, device_ids)
    
    # Convert model batchnorm layers to sync batchnorm layers
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Load model to device
    print(f"\nLoading model to device: {device}\n")
    model = model.to(device)

    # Distribute model across multiple GPUs
    print(f"\nDistributing model across GPU of rank: device_id = {rank}\n")
    model = DDP(model, device_ids=[rank])

    # # Load model to device
    # print(f"Loading model to device: {device}")
    # model.to(device)

    #####################################################################################
    # TRANSFORMS
    #####################################################################################
    # Set transforms
    if wandb.config.transforms == True:
        
        # Horizontal flip transform
        transform = transforms.functional.hflip
    else:
        transform = None
    
    print(f"Current hyperparameter values:\n {wandb.config}")

    #####################################################################################
    # DEFINE DATASET PATHS
    #####################################################################################
    # Define train, validation and output direcotry path
    data_dir, models_checkpoints_dir, train_paths, val_paths = define_dataset_paths(hpc=args.hpc_flag)

    #####################################################################################
    # DATALOADERS
    #####################################################################################
    # Define PyTorch datasets and dataloader

    # Define datasets
    train_dataset = KneeSegDataset3DMulticlass(train_paths, data_dir, num_classes=NUM_CLASSES, transform=transform)
    validation_dataset = KneeSegDataset3DMulticlass(val_paths, data_dir, num_classes=NUM_CLASSES, split='valid')

    # Define dataloaders
    # DistirbutedSampler - batch chunked over all gpus evently 
    # Shuffle = False required for DistributedSampler
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=int(batch_size), 
                                  #num_workers = 1, 
                                  sampler=DistributedSampler(train_dataset), 
                                  shuffle=False)
                                  
    validation_dataloader = DataLoader(validation_dataset, 
                                       batch_size=6, 
                                       #num_workers = 1, 
                                       sampler=DistributedSampler(validation_dataset), 
                                       shuffle=False)

    
    print(f"\n\n[GPU{rank}] train_dataloader length = {len(train_dataloader)}")
    print(f"[GPU{rank}] validation_dataloader length = {len(validation_dataloader)}\n\n")

    #####################################################################################
    # LOSS FUNCTION AND OPTIMISERS
    #####################################################################################
    # Specifiy criterion and optimiser
    loss_fn = create_loss(wandb.config.loss)
    # loss_fn = ce_dice_loss_multi_batch
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Removed weight decay for now as will address regularisation later if it's required

    # Removed for now
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True,
    #                                                           threshold=0.001, threshold_mode='abs')
    scaler = torch.amp.GradScaler('cuda')
    
    early_stopper = EarlyStopper(patience=5, min_delta=0.001)



    # Initialise minimum validation loss as infinity
    min_valid_loss = float('inf')

    # Model training
    print(f"TRAINING MODEL \n-------------------------------")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        train_loss, avg_train_dice, avg_train_haus, avg_train_dice_all, avg_train_haus_loss_all = train_loop(train_dataloader, device, model, loss_fn, optimizer, scaler, num_classes=NUM_CLASSES)
        valid_loss, avg_valid_dice, avg_valid_haus, avg_valid_dice_all, avg_valid_haus_loss_all = validation_loop(validation_dataloader, device, model, loss_fn, num_classes=NUM_CLASSES)

        if rank == 0:
            # log to wandb
            wandb.log({
                "Train Loss": train_loss, 
                "Train Dice Score": avg_train_dice,
                "Train Dice Score (Background)": avg_train_dice_all[0],
                "Train Dice Score (Femoral Cart.)": avg_train_dice_all[1],
                "Train Dice Score (Tibial Cart.)": avg_train_dice_all[2],
                "Train Dice Score (Patellar Cart.)": avg_train_dice_all[3],
                "Train Dice Score (Meniscus)": avg_train_dice_all[4],
                "Train Hausdorff Loss": avg_train_haus,
                "Train Hausdorff Loss (Background)": avg_train_haus_loss_all[0],
                "Train Hausdorff Loss (Femoral Cart.)": avg_train_haus_loss_all[1],
                "Train Hausdorff Loss (Tibial Cart.)": avg_train_haus_loss_all[2],
                "Train Hausdorff Loss (Patellar Cart.)": avg_train_haus_loss_all[3],
                "Train Hausdorff Loss (Meniscus)": avg_train_haus_loss_all[4],
                "Val Loss": valid_loss, 
                "Val Dice Score": avg_valid_dice,
                "Val Dice Score (Background)": avg_valid_dice_all[0],
                "Val Dice Score (Femoral Cart.)": avg_valid_dice_all[1],
                "Val Dice Score (Tibial Cart.)": avg_valid_dice_all[2],
                "Val Dice Score (Patellar Cart.)": avg_valid_dice_all[3],
                "Val Dice Score (Meniscus)": avg_valid_dice_all[4],
                "Val Hausdorff Loss": avg_valid_haus,
                "Val Hausdorff Loss (Background)": avg_valid_haus_loss_all[0],
                "Val Hausdorff Loss (Femoral Cart.)": avg_valid_haus_loss_all[1],
                "Val Hausdorff Loss (Tibial Cart.)": avg_valid_haus_loss_all[2],
                "Val Hausdorff Loss (Patellar Cart.)": avg_valid_haus_loss_all[3],
                "Val Hausdorff Loss (Meniscus)": avg_valid_haus_loss_all[4],
            })
        
        # Save as best if val loss is lowest so far
        # model.module.save required for DDP
        # Save checkpoint only for the rank 0 process
        if (valid_loss < min_valid_loss) and rank == 0:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            model_path = os.path.join(models_checkpoints_dir, f"{train_start_file}_{args.model}_multiclass_{run.name}_best_E.pth")
            torch.save(model.module.state_dict(), model_path)
            print(f"Best epoch yet: {epoch + 1}")
            
            # reset min as current
            min_valid_loss = valid_loss

        # Save model if early stopping triggered
        # model.module.save required for DDP
        # Save checkpoint only for the rank 0 process
        if early_stopper.early_stop(valid_loss) and rank == 0:
            print(f'Early stopping triggered! ({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            model_path = os.path.join(models_checkpoints_dir, f"{train_start_file}_{args.model}_multiclass_{run.name}_early_stop_E{epoch+1}.pth")
            torch.save(model.module.state_dict(), model_path)
            print(f"Early stop epoch: {epoch + 1}") 



    # Once training is done, save final model
    model_path = os.path.join(models_checkpoints_dir, f"{train_start_file}_{run.name}_final.pth")
    torch.save(model.state_dict(), model_path)

    # Cleanup - destroy process group to exit DDP training
    if rank == 0:
        wandb.finish()

    destroy_process_group()





def main_train(args) -> None:
    # Hyperparameter sweep configuration
    # config = wandb.config
    
    world_size = torch.cuda.device_count()

    # Set up distributed training
    # TODO review documentation for mp.spawn and the join argument
    mp.spawn(train, args=(world_size, wandb.config, args), nprocs=world_size, join=True)






if __name__ == '__main__':

    ###############################################################################################
    # PARSE COMMAND LINE ARGUMENTS
    ###############################################################################################

    # Parse command line arguments for:
    # Which model to train
    # Whether training is being done locally or on a HPC environment (this changes the data directory)
    # TODO - Take in model as argument (U-Net, MA-Net etc.), check it conforms to a set list
    # TODO - add dropout as hyperparameter - check how to do this in segmentation models library

    parser = argparse.ArgumentParser(
        prog="train",
        description="Trains an input model on the IWOAI OAI dataset subset",
        epilog=""
    )

    parser.add_argument('-hpc', '--hpc-flag', help='flag for whether program is run on locally or on hpc (hpc=1))') # add arg for hpc_flag
    parser.add_argument('-m', '--model') # add arg for model 
    args = parser.parse_args()

    print(f"Command line args = {args}")

    # Set running environment (True for HPC, False for local)
    HPC_FLAG = args.hpc_flag

    print(f"HPC_FLAG = {args.hpc_flag}")
    print(f"Model = {args.model}")



    #####################################################################################
    # SET HYPERPARAMETERS - PARSE STANRDARD INPUT (JSON)
    #####################################################################################

    # # Load in hyperparams using model CLI argument
    # config_filepath = os.path.join('.', 'config', f'config_{args.model}.json')

    # with open(config_filepath) as f:
    #     sweep_configuration = json.load(f)

    # print(f"sweep_configuation = {sweep_configuration}")

    std_input_data = None

    # If there is std input, read it into a variable
    if not sys.stdin.isatty(): 
        std_input_data = sys.stdin.read()
        print(f"Data from standard input:\n{std_input_data}")

    # If standard input data is present parse it as JSON data
    if std_input_data:
        try:
            sweep_configuration = json.loads(std_input_data)
            print(f"Parsed json hyperparameter config:\n{sweep_configuration}")
        
        except json.JSONDecodeError as e:
            print(f"Error reading JSON data from standard input:\n{e}")


    # Launch WandB sweep using hyperparameter sweep config
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="oai-subset-knee-cart-seg")

    wandb.agent(sweep_id, function=lambda: main_train(args))
    
