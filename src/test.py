
# %% Import libraries
import os
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from models.create_model import create_model
import datetime
import glob
import pandas as pd

from data.datasets import KneeSegDataset3DMulticlass

import json


# TODO: tidy up create model function
# TODO: save data as 3D numpy array in folder set by model name and date

def main(args):

    # %% Save run start time for output directory
    run_start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Run start time: {run_start_time}")

    # Create output directory
    if args.model == 'nnunet':
        pred_masks_dir = "/mnt/scratch/scjb/data/processed/oai_subset_knee_cart_seg/pred_masks/nnunet/postprocesing/"
        test_img_dir = "/mnt/scratch/scjb/nnUNet_raw/Dataset014_OAISubset/imagesTs/"
        test_img_paths = np.array([os.path.basename(i)[:7] for i in glob.glob(f'{test_img_dir}/*.nii.gz')])

        print(f'Number of test images: {len(test_img_paths)}')
        print(f'Test images: {test_img_paths}')
    else:
        pred_masks_dir = "/mnt/scratch/scjb/data/processed/oai_subset_knee_cart_seg/pred_masks"
        pred_masks_dir = os.path.join(pred_masks_dir, args.model, run_start_time)
        
        # Create output directory
        pred_masks_dir = Path(pred_masks_dir)
        pred_masks_dir.mkdir(parents=True, exist_ok=True)

        test_img_dir = os.path.join(args.data_dir, 'test')
        test_img_paths = np.array([os.path.basename(i).split('.')[0] for i in glob.glob(f'{test_img_dir}, *.npy')])
        print(f'Number of test images: {len(test_img_paths)}')
        print(f'Test images: {test_img_paths}')


    # If the model is not nnunet, create the model, load the weights and run inference
    if args.model != 'nnunet':
        # %% Read config json file into config variable
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # %% Create model using wandb config hyperparams
        model = create_model(input_model_arg=args.model, 
                            in_channels =config.in_channels, 
                            out_channels=config.out_channels, 
                            num_kernels= config.num_kernels, 
                            encoder=config.encoder, # None/null used in config file if not relevant for model
                            encoder_depth=config.encoder_depth, # None/null used if not relevant for model
                            img_size=config.img_size,
                            img_crop=config.img_crop,
                            feature_size=config.feature_size
        )

        model.load_state_dict(torch.load(args.model_weights))
    
        # Set model to evaluation mode and move to device
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create dataset and dataloader


        test_dataset = KneeSegDataset3DMulticlass(data_dir=args.data_dir, split='test', img_crop=config.img_crop)    
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Process each image
        with torch.no_grad():

            for idx, (im, mask) in enumerate(test_loader):
                
                # Load X and y to releavnt device
                im = im.to(device)
                mask = mask.to(device)

                # Forward pass
                pred = model(im)

                # Save prediction
                pred = pred.cpu().numpy()

                print(f"Prediction shape: {pred.shape}\n")

                # convert model outputs from logits to probability
                pred_prob = nn.functional.softmax(pred, dim=1)

                pred_binary_mask = (pred>0.5).astype(int)

                # save predicted mask
                np.save(os.path.join(pred_masks_dir, test_img_paths[idx]), pred_binary_mask)


    # Create list of predicted segentation masks - nnunet outputs .nii.gz whereas other models output .npy
    if args.model != 'nnunet':
        pred_masks = [os.path.join(pred_masks_dir, i) for i in os.listdir(pred_masks_dir) if i.endswith('.npy')]
    else:
        pred_masks = [os.path.join(pred_masks_dir, i) for i in os.listdir(pred_masks_dir) if i.endswith('.nii.gz')]

    print(f"Number of predicted masks: {len(pred_masks)}")
    print(f"Predicted masks: {pred_masks}")



    # Initialise lists to store evaluation metrics
    dice_scores = []
    hausdorff_distances = []
    assd = []
    voe = []
    te = []

    # Get list of base filenames for each mask
    mask_base_filenames = [os.path.basename(i).split('.')[0] for i in pred_masks]

    # Loop through each predicted mask and save as nifti file
    for mask in pred_masks:
        pass

        # # Load mask
        # mask = np.load(mask)

        # # Save mask as nifti file
        # mask = nib.Nifti1Image(mask, np.eye(4))
        # nib.save(mask, os.path.join(pred_masks_dir, f"{os.path.basename(mask).split('.')[0]}.nii.gz"))


        # Dice Score
        # dice = dice_score(mask, y)
        # print(f"Dice score: {dice}")
        # Save to dice score list


        # Hausdorff distance
        # hausdorff = hausdorff_distance(mask, y)
        # print(f"Hausdorff distance: {hausdorff}")
        # Save to hausdorff distance list


        # Average symmetric surface distance
        # assd = average_symmetric_surface_distance(mask, y)
        # print(f"Average symmetric surface distance: {assd}")
        # Save to assd list

        # Volmeetric Overlap Error
        # voe = volumetric_overlap_error(mask, y)
        # print(f"Volumetric Overlap Error: {voe}")
        # Save to voe list

        # Thickness error
        # te = thickness_error(mask, y)
        # print(f"Thickness error: {te}")
        # Save to te list



    # Combine evaluation metrics into a pandas dataframe 

    # eval_metrics = pd.DataFrame({'dice': dice_scores, 
    #                              'hausdorff': hausdorff_distances, 
    #                              'assd': assd, 
    #                              'voe': voe, 
    #                              'te': te})

    # # Save evaluation metrics as csv
    # eval_metrics.to_csv(os.path.join(pred_masks_dir, 'eval_metrics.csv'))


if __name__ == '__main__':
# %%
 # %% Read in json config file as command line argument using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Model architecture')
    parser.add_argument('--model_weights', type=str, required=False, help='Model weights')
    parser.add_argument('--data_dir', type=str, required=False, default="/mnt/scratch/scjb/data/oai_subset/", help='Path to test data')

    args = parser.parse_args()

    main(args)
