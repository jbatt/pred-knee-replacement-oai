
# %% Import libraries
import os
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import datetime
import glob
import pandas as pd
import monai
from data.datasets import KneeSegDataset3DMulticlass
import json
from skimage import morphology

from models.create_model import create_model
from utils.utils import crop_mask
from metrics.metrics import calculate_mean_thickness

from monai.inferers import sliding_window_inference

# TODO: tidy up create model function
# TODO: save data as 3D numpy array in folder set by model name and date

def main(args):

    NUM_CLASSES = 5
    # %% Save run start time for output directory
    run_start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Run start time: {run_start_time}")
    print(f"Args: {args}\n\n")

    res_dir = "/users/scjb/pred-knee-replacement-oai/results/eval_metrics"
    res_dir = Path(os.path.join(res_dir, args.model, run_start_time))
    
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
        print(f"Pred masks dir: {pred_masks_dir}")



    print(f"args.data_dir: {args.data_dir}")
    test_gt_dir = os.path.join(args.data_dir, 'test_gt')
    test_gt_paths = [i for i in glob.glob(f'{test_gt_dir}/*.npy')]
    test_gt_paths = sorted(test_gt_paths)



    print(f'Number of test images: {len(test_gt_paths)}')
    print(f'Test images: {test_gt_paths}')

    # %% Read config json file into config variable
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print(f"Config file: {config}\n\n")

    # If the model is not nnunet, create the model, load the weights and run inference
    if args.inference:

        test_dir = os.path.join(args.data_dir, 'test')
        test_img_paths = [os.path.basename(i).split('.')[0] for i in glob.glob(f'{test_dir}/*.im')]
        test_img_paths = sorted(test_img_paths)

        print(f'Number of test images: {len(test_img_paths)}')
        print(f'Test images: {test_img_paths}\n\n')



        # %% Create model using wandb config hyperparams


        print(f"""Creating model: {args.model}
              Model parameters:
                in_channels: {config['parameters']['in_channels']['values']}
                out_channels: {config['parameters']['out_channels']['values']}
                num_kernels: {config['parameters']['num_kernels']['values']}
                encoder: {config['parameters']['encoder']['values']}
                encoder_depth: {config['parameters']['encoder_depth']['values']}
                patch_size: {config['parameters']['patch']['parameters']['patch_size']['values']}
                num_heads: {config['parameters']['transformer']['parameters']['num_heads']['values']}
                depths: {config['parameters']['transformer']['parameters']['depths']['values']}\n\n
        """)


        model = create_model(input_model_arg=args.model,
                    in_channels =config['parameters']['in_channels']['values'][0], 
                    out_channels=config['parameters']['out_channels']['values'][0], 
                    num_kernels= config['parameters']['num_kernels']['values'][0], 
                    encoder=config['parameters']['encoder']['values'][0], # None/null used in config file if not relevant for model
                    encoder_depth=config['parameters']['encoder_depth']['values'][0], # None/null used if not relevant for model
                    patch_size=config['parameters']['patch']['parameters']['patch_size']['values'][0],
                    num_heads=config['parameters']['transformer']['parameters']['num_heads']['values'][0],
                    depths=config['parameters']['transformer']['parameters']['depths']['values'][0]
        )

        model.load_state_dict(torch.load(args.model_weights))
    
        # Set model to evaluation mode and move to device
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create dataset and dataloader
        test_dataset = KneeSegDataset3DMulticlass(file_paths=test_img_paths, 
                                                  data_dir=args.data_dir, 
                                                  num_classes=NUM_CLASSES, 
                                                  split='test', 
                                                  img_crop=config['parameters']['img_crop']['values'][0])    


        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Process each image
        with torch.no_grad():

            for idx, (im, mask) in enumerate(test_loader):
                
                # Load X and y to releavnt device
                im = im.to(device)
                mask = mask.to(device)

                # Forward pass
                pred = model(im)

                # If using a transformer-based model, use sliding window inference
                if args.model in ['swin_unetr', 'segformer3d']:
                                
                    print(f"Performing sliding window inference with patch size: {config['parameters']['patch']['parameters']['patch_size']['values'][0]}")
                    # Perform sliding window inference
                    pred = sliding_window_inference(im, 
                                                    roi_size=config['parameters']['patch']['parameters']['patch_size']['values'][0], 
                                                    sw_batch_size=4, 
                                                    predictor=model, 
                                                    overlap=config['parameters']['patch']['parameters']['inference_overlap']['values'][0])
                    
                    print(f"Prediction shape from sliding window: {pred.shape}")
                        
                else:
                    # Make predictions on the input features
                    pred = model(im)


    
                print(f"Prediction shape: {pred.shape}\n")
                print(f"Prediction type: {type(pred)}")

                # convert model outputs from logits to probability
                pred_prob = nn.functional.softmax(pred, dim=1)

                # Get predicted class
                pred_binary_mask = torch.argmax(pred_prob, dim=1)

                # Save prediction
                pred_binary_mask = pred_binary_mask.cpu().numpy()

                # save predicted mask
                print(f"Saving predicted mask {os.path.basename(test_gt_paths[idx])} to {pred_masks_dir}")
                np.save(os.path.join(pred_masks_dir, os.path.basename(test_gt_paths[idx])), pred_binary_mask)


    # Create list of predicted segentation masks - nnunet outputs .nii.gz whereas other models output .npy
    if args.model != 'nnunet':
        pred_mask_paths = [os.path.join(pred_masks_dir, os.path.basename(i)) for i in os.listdir(pred_masks_dir) if i.endswith('.npy')]
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
    assds = []
    voes = []
    cvs = []
    y_thickness_means = []
    thickness_errors = []
    
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
            y_pred = y_pred.astype(np.int16)
            y_pred_nii = nib.Nifti1Image(y_pred, np.eye(4)) # changed to y_pred from "mask"
            nib.save(y_pred_nii, os.path.join(pred_masks_dir, f"{os.path.basename(pred_mask_path).split('.')[0]}.nii.gz"))

        y = np.load(gt_im_path)
        print(f"y shape pre-cropping: {y.shape}")
        print(f"y unique values: {np.unique(y)}")

        # Move classes dimension to be first dimension
        y = np.transpose(y, (3,0,1,2)) # TODO: this may be needed for nnunet
        
        # Crop ground truth to match predicted
        # if args.model == 'nnunet':
        #     y = crop_mask(y, dim1_lower=40, dim1_upper=312, dim2_lower=42, dim2_upper=314, onehot=True)

        
        print(f""" Crop parameters:
            dim1_lower = {config['parameters']['img_crop']['values'][0][0][0]}
            dim1_upper = {config['parameters']['img_crop']['values'][0][0][1]}
            dim2_lower = {config['parameters']['img_crop']['values'][0][1][0]}
            dim2_upper = {config['parameters']['img_crop']['values'][0][1][1]}
        """)

        y = crop_mask(y, 
                      dim1_lower=config['parameters']['img_crop']['values'][0][0][0], 
                      dim1_upper=config['parameters']['img_crop']['values'][0][0][1], 
                      dim2_lower=config['parameters']['img_crop']['values'][0][1][0], 
                      dim2_upper=config['parameters']['img_crop']['values'][0][1][1])


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
        
        print(f"y_pred shape pre unsqueeze: {y_pred.shape}")
        y_pred = torch.unsqueeze(torch.tensor(y_pred), dim=0)
        
        if args.model == 'nnunet': # if using nnunet, add a second dim of 1
            y_pred = torch.unsqueeze(torch.tensor(y_pred), dim=0)
    
        print(f"y_pred shape post unsqueeze: {y_pred.shape}")


        # Convert y_pred to onehot encoding
        y_pred = monai.networks.utils.one_hot(y_pred, num_classes=5)

        print(f"y_pred shape: {y_pred.shape}\ny_pred type: {type(y_pred)}\ny_pred values: {np.unique(y_pred)}")
        print(f"y shape: {y.shape}\ny type: {type(y)}\ny values: {np.unique(y)}")
        
        
        y = torch.tensor(y)
        y_pred = torch.tensor(y_pred)

        # Dice Score
        # Calculate Dice score for each tissue
        dice = monai.metrics.DiceHelper(include_background=False)(y_pred, y)
        print(f"\n\nDice score for {gt_im_path}: {dice}")
        # Remove dim of 1 and conert to list
        dice = dice[0].squeeze().tolist()
        # Append file name to list (first column of output df)
        dice.append(os.path.basename(gt_im_path))
        # Append Dice score for each tissue
        dice_scores.append(dice)
        print(f"dice scores: {dice_scores}")

        # Hausdorff distance # TODO: implement spacing in HD calc?
        hd = monai.metrics.compute_hausdorff_distance(y_pred, y, include_background=False, spacing=[0.36,0.36,0.7])
        hd = hd.squeeze().tolist()
        hd.append(os.path.basename(gt_im_path))
        print(f"Hausdorff distance for {gt_im_path}: {hd}")
        hausdorff_distances.append(hd)

        # Average symmetric surface distance (ASSD)
        assd = monai.metrics.compute_average_surface_distance(y_pred, y, symmetric=True, include_background=False, spacing=[0.36,0.36,0.7])
        assd = assd.squeeze().tolist()
        assd.append(os.path.basename(gt_im_path))
        print(f"Average symmetric surface distance ofr {gt_im_path}: {assd}")
        assds.append(assd)

        # Volmetric Overlap Error (VOE)
        iou = monai.metrics.compute_iou(y_pred, y, include_background=False)
        print(f"IOU: {iou}")
        voe = 1 - iou
        voe = voe.squeeze().tolist()
        voe.append(os.path.basename(gt_im_path))
        print(f"Volumetric overlap error {gt_im_path}: {voe}")
        voes.append(voe)

        # Coefficient of variation
        # TODO: decide approach to calculate coefficent of variation


        # Thickness error
        y_thickness = calculate_mean_thickness(y)
        y_pred_thickness = calculate_mean_thickness(y_pred) 
        thickness_error = np.array(y_pred_thickness) - np.array(y_thickness)
        thickness_error = thickness_error.tolist()
        thickness_error.append(os.path.basename(gt_im_path))


        # Thickness mean
        y_thickness_mean = calculate_mean_thickness(y)
        y_thickness_mean.append(os.path.basename(gt_im_path))

        print(f"Thickness mean for {gt_im_path}: {y_thickness_mean}")
        y_thickness_means.append(y_thickness_mean)




    # Combine evaluation metrics into a pandas dataframe 

    df_dice = pd.DataFrame(dice_scores, columns=["fem cart.", 
                                                 "tibial cart.", 
                                                 "patellar cart.", 
                                                 "meniscus", 
                                                 "img"])
    df_hd = pd.DataFrame(hausdorff_distances, columns=["fem cart.", 
                                                 "tibial cart.", 
                                                 "patellar cart.", 
                                                 "meniscus", 
                                                 "img"])
    df_assd = pd.DataFrame(assds, columns=["fem cart.", 
                                                 "tibial cart.", 
                                                 "patellar cart.", 
                                                 "meniscus", 
                                                 "img"])
    df_voe = pd.DataFrame(voes, columns=["fem cart.", 
                                                 "tibial cart.", 
                                                 "patellar cart.", 
                                                 "meniscus", 
                                                 "img"])

    df_te = pd.DataFrame(thickness_errors, columns=["fem cart.", 
                                                 "tibial cart.", 
                                                 "patellar cart.", 
                                                 "meniscus", 
                                                 "img"])
    df_tm = pd.DataFrame(y_thickness_means, columns=["fem cart.", 
                                                "tibial cart.", 
                                                "patellar cart.", 
                                                "meniscus", 
                                                "img"])
    
    # Make results directory
    res_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluation metrics as csv
    df_dice.to_csv(os.path.join(res_dir, f'dice_{args.model}_{run_start_time}.csv'))
    df_hd.to_csv(os.path.join(res_dir, f'hd_{args.model}_{run_start_time}.csv'))
    df_assd.to_csv(os.path.join(res_dir, f'assd_{args.model}_{run_start_time}.csv'))
    df_voe.to_csv(os.path.join(res_dir, f'voe_{args.model}_{run_start_time}.csv'))
    df_voe.to_csv(os.path.join(res_dir, f'te_{args.model}_{run_start_time}.csv'))
    df_tm.to_csv(os.path.join(res_dir, f'tm_{args.model}_{run_start_time}.csv'))


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
