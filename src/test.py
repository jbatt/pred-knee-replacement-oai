
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
import monai
from data.datasets import KneeSegDataset3DMulticlass
import h5py
import json
from utils.utils import crop_mask


# TODO: tidy up create model function
# TODO: save data as 3D numpy array in folder set by model name and date

def main(args):

    # %% Save run start time for output directory
    run_start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Run start time: {run_start_time}")

    # Create output directory
    if args.model == 'nnunet':
        pred_masks_dir = "/mnt/scratch/scjb/data/processed/oai_subset_knee_cart_seg/pred_masks/nnunet/postprocesing/"
        
        # TODO: Update test images to be the original test ground truths 
        # test_img_dir = "/mnt/scratch/scjb/nnUNet_raw/Dataset014_OAISubset/imagesTs/"
        # test_img_paths = [i for i in glob.glob(f'{test_img_dir}/*.nii.gz')]
        # test_img_paths = sorted(test_img_paths)

    else:
        pred_masks_dir = "/mnt/scratch/scjb/data/processed/oai_subset_knee_cart_seg/pred_masks"

        # If running inference create new folder for the new predicted masks
        if args.inference:
            pred_masks_dir = os.path.join(pred_masks_dir, args.model, run_start_time)
        else:
            pred_masks_dir = os.path.join(pred_masks_dir, args.model)

        # Create output directory
        pred_masks_dir = Path(pred_masks_dir)
        pred_masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"args.data_dir: {args.data_dir}")
    test_gt_dir = os.path.join(args.data_dir, 'test_gt')
    test_gt_paths = [i for i in glob.glob(f'{test_gt_dir}/*.npy')]
    test_gt_paths = sorted(test_gt_paths)

    print(f'Number of test images: {len(test_gt_paths)}')
    print(f'Test images: {test_gt_paths}')


    # If the model is not nnunet, create the model, load the weights and run inference
    if args.inference:
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
                np.save(os.path.join(pred_masks_dir, test_gt_paths[idx]), pred_binary_mask)


    # Create list of predicted segentation masks - nnunet outputs .nii.gz whereas other models output .npy
    if args.model != 'nnunet':
        pred_mask_paths = [os.path.join(pred_masks_dir, i) for i in os.listdir(pred_masks_dir) if i.endswith('.npy')]
        pred_mask_paths = sorted(pred_mask_paths)
        
    else:
        pred_mask_paths = [os.path.join(pred_masks_dir, i) for i in os.listdir(pred_masks_dir) if i.endswith('.nii.gz')]
        pred_mask_paths = sorted(pred_mask_paths)

    print(f"Number of predicted masks: {len(pred_mask_paths)}")
    print(f"Predicted masks: {pred_mask_paths}")


    # pred_masks_paths = np.array([os.path.basename(i) for i in glob.glob(f'{pred_masks_dir}/*.npy')])
    # pred_masks_paths = sorted(pred_masks_paths)
    
    # print(f"Number of predicted masks {len(pred_masks_paths)}")
    # print(f"Predicted masks: {pred_masks_paths}")
    

    # Initialise lists to store evaluation metrics
    dice_scores = []
    hausdorff_distances = []
    assd = []
    voe = []
    te = []

    # Get list of base filenames for each mask

    # Loop through each predicted mask and save as nifti file
    for gt_im_path, pred_mask_path in zip(test_gt_paths, pred_mask_paths):
        
        print(f"\n------------------------------------\n")
        print(f"Current groud truth: {gt_im_path}")
        print(f"Current predicted mask: {pred_mask_path}\n\n")

        # Load mask
        if args.model == "nnunet":
            y_pred = nib.load(pred_mask_path).get_fdata()
            # y = nib.load(gt_im_path).get_fdata()
        
        else:
            y_pred = np.load(pred_mask_path)
            # Save mask as nifti file as useful for plotting later
            y_pred_nii = nib.Nifti1Image(mask, np.eye(4))
            nib.save(y_pred_nii, os.path.join(pred_masks_dir, f"{os.path.basename(pred_mask_path).split('.')[0]}.nii.gz"))

        y = np.load(gt_im_path)

        # Move classes dimension to be firt dimension
        y = np.transpose(y, (3,0,1,2))
        
        # Crop ground truth to match predicted
        y = crop_mask(y, dim1_lower=40, dim1_upper=312, dim2_lower=42, dim2_upper=314, onehot=True)



        # Add background channel to ground truth - y
        # Add background to mask - if everything in a position is zero, it's a background voxel
        y_all_classes_zero = np.all(y == 0, axis=0)

        # Set background masks to be intergers
        y_bg_mask = y_all_classes_zero.astype(int)

        # Add dimension of ones to enable concatenation
        y_bg_mask = np.expand_dims(y_bg_mask, axis=0)

        # Concatenate background to 4-class mask (background first, then 4 tissue types)
        y = np.concatenate([y_bg_mask, y], axis=0)
        
        # Add batch of 1 to y and y_pred for monai dice calc
        y = torch.unsqueeze(torch.tensor(y), dim=0)
        
        y_pred = torch.unsqueeze(torch.tensor(y_pred), dim=0)
        y_pred = torch.unsqueeze(torch.tensor(y_pred), dim=0)

        # Convert y_pred to onehot encoding
        y_pred = monai.networks.utils.one_hot(y_pred, num_classes=5)

        print(f"y_pred shape: {y_pred.shape}\ny_pred type: {type(y_pred)}\ny_pred values: {np.unique(y_pred)}")
        print(f"y shape: {y.shape}\ny type: {type(y)}\ny values: {np.unique(y)}")
        
        
        # Dice Score
        # Save to dice score list

        dice = monai.metrics.DiceHelper(include_background=True)(torch.tensor(y_pred), torch.tensor(y))
        print(f"\n\nDice score for {gt_im_path}: {dice}\n\n")

        # Test comment

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
    parser.add_argument('--inference', action=argparse.BooleanOptionalAction, help='Whether or not to run inference')

    args = parser.parse_args()

    main(args)
