import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import glob
import os
import sys

# Include src directory in path to import custom modules
if '../src' not in sys.path:
    sys.path.append('../src')

from models.model_unet import UNet3D, UNet3DMulticlass
import models.evaluation
from models.evaluation import dice_coefficient_batch, dice_coefficient, dice_coefficient_multi_batch_all

import data.datasets
from data.datasets import KneeSegDataset3D, KneeSegDataset3DMulticlass


# Set running environment (True for HPC, False for local)
HPC_FLAG = sys.argv[1]
HPC_FLAG

weight_filename = sys.argv[2]

if HPC_FLAG == "1":
    # Define data directory - ARC4
    # Using absolute paths because ray-tune changing working directory
    DATA_DIRECTORY = '/nobackup/scjb/data/oai_subset'
    DATA_TRAIN_DIRECTORY = '/nobackup/scjb/data/oai_subset/train'
    DATA_VALID_DIRECTORY = '/nobackup/scjb/data/oai_subset/valid'
    DATA_TEST_DIRECTORY = '/nobackup/scjb/data/oai_subset/test'
    DATA_PROCESSED_DIRECTORY = '/home/home02/scjb/pred-knee-replacement-oai/data/processed'
    RESULTS_PATH = '/home/home02/scjb/pred-knee-replacement-oai/results'
    MODELS_PATH = '/home/home02/scjb/pred-knee-replacement-oai/models'
    MODELS_CHECKPOINTS_PATH = '/home/home02/scjb/pred-knee-replacement-oai/models/checkpoints'
    RESULTS_PATH = '/home/home02/scjb/pred-knee-replacement-oai/results'
    

else:
    # Define data directory for local runs
    DATA_DIRECTORY = 'C:\\Users\\james\\OneDrive - University of Leeds\\1. Projects\\1.1 PhD\\1.1.1 Project\\Data\\OAI Subset\\'
    DATA_TRAIN_DIRECTORY = 'C:\\Users\\james\\OneDrive - University of Leeds\\1. Projects\\1.1 PhD\\1.1.1 Project\\Data\\OAI Subset\\train'
    DATA_VALID_DIRECTORY = 'C:\\Users\\james\\OneDrive - University of Leeds\\1. Projects\\1.1 PhD\\1.1.1 Project\\Data\\OAI Subset\\valid'
    DATA_VALID_DIRECTORY = 'C:\\Users\\james\\OneDrive - University of Leeds\\1. Projects\\1.1 PhD\\1.1.1 Project\\Data\\OAI Subset\\test'

    RESULTS_PATH = '../results'
    MODELS_PATH = '../models'
    MODELS_CHECKPOINTS_PATH = '../models/checkpoints'
    RESULTS_PATH = '../results/'


# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")


# Define test file paths
test_paths = [os.path.basename(x).split(".")[0] for x in glob.glob(os.path.join(DATA_TEST_DIRECTORY, "*"))]
print(f"length of test_paths: {len(test_paths)}")
print(f"test_paths first entry: {test_paths[0]}")


# Create test dataset and dataloader
test_multi_dataset = KneeSegDataset3DMulticlass(test_paths, DATA_DIRECTORY, split='test', num_classes=5)
test_multi_loader = DataLoader(test_multi_dataset, batch_size=1, shuffle=False)

# Create model
model = UNet3DMulticlass(1, 5, 16)


# Load model weights
# weight_filename = "2024-09-03-01_17_27703401_high_lr_40_epoch_final.pth" - going to pass in as parameter
weights_path = os.path.join(MODELS_CHECKPOINTS_PATH, weight_filename)
multi_model_weights = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)

# load weights into model
model.load_state_dict(multi_model_weights)

# set model to eval mode
model.eval

# load model to device
model.to(device)

dice_scores = []
dice_scores_all_classes = []

# loop through testloader (use tqdm)
for idx, (im, gt_mask) in enumerate(test_multi_loader):

    # 	load im to device
    im.to(device)
    print(f"Image shape: {im.shape}")

    gt_mask = gt_mask.squeeze(axis=1)
    gt_mask.to(device)
    print(f"Mask shape: {gt_mask.shape}\n")

    # make prediction using model
    pred = model(im)
    print(f"Prediction shape: {pred.shape}\n")

    pred_binary_mask = (pred>0.5).int()

    # calculate dice coefficient
    dice_score_all_classes = dice_coefficient_multi_batch_all(gt_mask, pred_binary_mask, num_labels=5).detach()
    dice_scores_all_classes.append(dice_score_all_classes.tolist())

    dice_score = dice_score_all_classes.mean()
    dice_scores.append(dice_score.item())
    
    print(f"Im {idx+1} ({test_paths[idx]}): Dice = {dice_score}")
    print(f"Im {idx+1} ({test_paths[idx]}): Dice by call = {dice_score_all_classes}")

    # save predicted mask
    np.save(os.path.join(DATA_PROCESSED_DIRECTORY, test_paths[idx]), pred_binary_mask)


dice_scores_all_classes = np.array(dice_scores_all_classes)
dice_scores = np.array(dice_scores)

# np.save(os.path.join(RESULTS_PATH, "dice_scores_mean"), pred_binary_mask)

print(f"Mean dice score: {dice_scores.mean()}")
print(f"Mean dice score for each class: {dice_scores.mean(axis=0)}")