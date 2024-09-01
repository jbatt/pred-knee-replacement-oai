"""
File where all datasets are defined.
For the 3D U-Net, the 3D dataset class was used, where the full images and masks were loaded.

"""

import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
import os
import numpy as np
from utils.utils import crop_im, crop_mask, clip_and_norm, pad_to_square


# Define the 3D Dataset class
# Image and mask both need same transforms to be applied, so DO NOT USE RANDOM TRANSFORMS
# - use e.g. transforms.functional.hflip which has no randomness.

class KneeSegDataset3D(Dataset):
    def __init__(self, file_paths, data_dir, split='train', transform=None, transform_chance=0.5):
        self.file_paths = file_paths
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.transform_chance = transform_chance

    # Return length of dataset
    def __len__(self):
        return len(self.file_paths)

    # Get an item from the dataset
    def __getitem__(self, index):
        
        path = self.file_paths[index]

        # Test data is arranged differently, and as mask is numpy as opposed to h5py
        # Load image and segmentation mask from test data (numpy arrays)
        if self.split == 'test':
            im_path = os.path.join(self.data_dir, self.split, path + '.im')
            seg_path = os.path.join(self.data_dir, 'test_gt', path + '.npy')
            with h5py.File(im_path,'r') as hf:
                image = np.array(hf['data'])
            mask = np.load(seg_path)

        # Train and Validation h5py files
        # Extract image and mask from training and validation data
        else: 
            # get full paths and read in
            im_path = os.path.join(self.data_dir, self.split, path + '.im')
            seg_path = os.path.join(self.data_dir, self.split, path + '.seg')
            with h5py.File(im_path,'r') as hf:
                image = np.array(hf['data'])
            with h5py.File(seg_path,'r') as hf:
                mask = np.array(hf['data'])


        # Extract the meniscus mask
        
        # TODO: extract all masks 
        if self.split == 'test':
            minisc_mask = mask[...,-1]
        else:

            # medial meniscus
            med_mask = mask[...,-1]

            # THERE IS ONE ERRANT CASE IN TRAIN SET. LATERAL MENISCUS IS AT WRONG INDEX
            # lateral
            if path == 'train_026_V01':
                lat_mask = mask[...,2]
            else:
                lat_mask = mask[...,-2]

            # both together
            minisc_mask = np.add(med_mask,lat_mask)

        mask = np.clip(minisc_mask, 0, 1) #just incase the two menisci ground truths overlap, clip at 1

        # TODO: update cropping to include all masks
        # crop image/mask
        image = crop_im(image, dim1_lower=120, dim1_upper=320, dim2_lower=70, dim2_upper=326)
        mask = crop_im(mask, dim1_lower=120, dim1_upper=320, dim2_lower=70, dim2_upper=326)

        # normalise image
        image = clip_and_norm(image, 0.005)

        # turn to torch, add channel dimension, and return
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        # transforms?
        if self.transform != None:

            # Manually apply trasnforms with randomisation as image and mask must have the same transforms applied
            # Generate a random number, if above a threshold apply the transform to the image and the mask 
            if random.random() < self.transform_chance:
                image = self.transform(image)
                mask = self.transform(mask)

        return image, mask


# Multiclass knee cartiage segmentation masks
class KneeSegDataset3DMulticlass(Dataset):
    def __init__(self, file_paths, data_dir, num_classes, split='train', transform=None, transform_chance=0.5):
        self.file_paths = file_paths
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.transform_chance = transform_chance
        self.num_classes = num_classes

    # Return length of dataset
    def __len__(self):
        return len(self.file_paths)

    # Get an item from the dataset
    def __getitem__(self, index):
        
        path = self.file_paths[index]

        # Test data is arranged differently, and as mask is numpy as opposed to h5py
        # Load image and segmentation mask from test data (numpy arrays)
        if self.split == 'test':
            im_path = os.path.join(self.data_dir, self.split, path + '.im')
            seg_path = os.path.join(self.data_dir, 'test_gt', path + '.npy')

            # Open the image file
            with h5py.File(im_path,'r') as hf:
                image = np.array(hf['data'])
            
            # Load the mask
            mask = np.load(seg_path)


        # Train and Validation h5py files
        # Extract image and mask from training and validation data
        else: 
            # get full paths and read in
            im_path = os.path.join(self.data_dir, self.split, path + '.im')
            seg_path = os.path.join(self.data_dir, self.split, path + '.seg')

            # Open the image file and load it to a numpy array
            with h5py.File(im_path,'r') as hf:
                image = np.array(hf['data'])
            
            # Open the mask file and load it to a numpy array
            with h5py.File(seg_path,'r') as hf:
                mask = np.array(hf['data'])

        # Extract the meniscus mask
        
        # TODO: extract all masks 
        
        # If test data - continue with all four classes
        if self.split == 'test':
            pass
        
        # If train or validation - combine the medial/lateral masks for the tibial cart. and meniscus 
        else:

            # Define medial meniscus mask
            menisc_med_mask = mask[...,-1]

            # Define lateral meniscus mask
            # Below captures single train example with lateral meniscus mask at wrong index
            if path == 'train_026_V01':
                menisc_lat_mask = mask[...,2]
            else:
                menisc_lat_mask = mask[...,-2]

            # both together
            minisc_mask = np.add(menisc_med_mask, menisc_lat_mask)

            # # Define tibial medial and lateral cart. masks
            # tib_med_mask = mask[...,1]
            # tib_lat_mask = mask[...,2]
            
            # # Combine tibial masks
            # tib_mask = np.add(tib_med_mask, tib_lat_mask)

            # # Define femoral cart. mask
            # fem_mask =  mask[...,0]

            # # Define femoral cart. mask
            # pat_mask =  mask[...,3]

            # # Create mask using combined tissue masks
            # combined_mask = np.stack([fem_mask, tib_mask, pat_mask, minisc_mask], axis=-1)

            tibial_mask = np.add(mask[:,:,:,1], mask[:,:,:,2])

            # Clip minsc and tibial masks at 1 in case the two ground truths overlap
            minisc_mask = np.clip(minisc_mask, 0, 1)
            tibial_mask = np.clip(tibial_mask, 0, 1)

            # # Create base mask of all zero using shape of mensicus mask
            # menisc_mask_shape = minisc_mask.shape
            # mask_all = np.zeros(menisc_mask_shape)

            # # Fill in class index values based on binary masks: 0=background, 1=femoral, 2=tibial, 3=patellar, 4=meniscus
            # mask_all[mask[:,:,:,0]==1] = 1
            # mask_all[tibial_mask[:,:,:]==1] = 2
            # mask_all[mask[:,:,:,3]==1] = 3
            # mask_all[minisc_mask[:,:,:]==1] = 4

            # Adjust mask dimension from 6 classes to 4 classes
            mask_dims = mask.shape[:-1]
            mask_dims += (self.num_classes,)
            mask_dims

            # Initalise mask
            mask_all = np.zeros(mask_dims)

            # Fill in each layer of multiclass mask with each classes seg mask
            mask_all[:,:,:,1] = mask[:,:,:,0]
            mask_all[:,:,:,2] = tibial_mask 
            mask_all[:,:,:,3] = mask[:,:,:,3]
            mask_all[:,:,:,4] = minisc_mask
            
            print(f"Dataset original gt mask dimension: {mask_all.shape}")

            # Change dimension order to match prediction output dimensions for loss function
            mask_all = mask_all.transpose(3,0,1,2)

            print(f"Dataset post-transpose gt mask dimension: {mask_all.shape}")

        # # Clip in case ground truths overlap
        # mask = np.clip(mask_all, 0, 1) 
        
        # crop image/mask
        image = crop_im(image, dim1_lower=72, dim1_upper=312, dim2_lower=74, dim2_upper=322)
        mask = crop_mask(mask_all, dim1_lower=72, dim1_upper=312, dim2_lower=74, dim2_upper=322)

        # normalise image
        image = clip_and_norm(image, 0.005)

        # turn to torch, add channel dimension, and return
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        # transforms?
        if self.transform != None:

            # Manually apply trasnforms with randomisation as image and mask must have the same transforms applied
            # Generate a random number, if above a threshold apply the transform to the image and the mask 
            if random.random() < self.transform_chance:
                image = self.transform(image)
                mask = self.transform(mask)

        return image, mask