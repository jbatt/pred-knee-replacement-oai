import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
import os
import numpy as np
from utils.utils import crop_im, crop_mask, clip_and_norm


# Multiclass knee cartilage MRI volumes and segmentation masks
class KneeSegDataset3DMulticlass(Dataset):
    def __init__(self, file_paths, data_dir, num_classes, img_crop=[[25,271],[25,271],[0,160]], split='train', transform=None, transform_chance=0.5):
        self.file_paths = file_paths
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.transform_chance = transform_chance
        self.num_classes = num_classes
        self.img_crop = img_crop

    # Return length of dataset
    def __len__(self):
        return len(self.file_paths)

    # Get an item from the dataset
    def __getitem__(self, index):
        
        path = self.file_paths[index]

        # Test data are saved as numpy as instead of h5py

        # Load image and segmentation mask from test data (numpy arrays)
        if self.split == 'test':
            im_path = os.path.join(self.data_dir, self.split, path + '.im')
            seg_path = os.path.join(self.data_dir, 'test_gt', path + '.npy')

            # Open the image file
            with h5py.File(im_path,'r') as hf:
                image = np.array(hf['data'])
            
            # Load the mask
            mask = np.load(seg_path)


        # Train and validation data are h5py files

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

        

        # Manipulate masks so there are 5 masks in the right order: 
            # Masks: background, femoral cart., tibial cart., patellar cart., meniscus   
        
        # If test data (test data already has just four classes) - continue with all four classes
        if self.split == 'test':
            
            # Define background mask for the test data

            # Reorder dimensions to match model
            mask = mask.transpose(3,0,1,2)

            # Add background to mask - if everything in a position is zero, it's a background voxel
            all_classes_zero_mask = np.all(mask == 0, axis=0)

            # Set background masks to be intergers
            background_mask = all_classes_zero_mask.astype(int)

            # Add dimension of ones to enable concatenation
            background_mask = np.expand_dims(background_mask, axis=0)

            # Concatenate background to 4-class mask (background first, then 4 tissue types)
            mask = np.concatenate([background_mask, mask], axis=0)
        

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

            # Combine lateral and medial meniscus masks
            minisc_mask = np.add(menisc_med_mask, menisc_lat_mask)

            # Combine medial and lateral tibial masks
            tibial_mask = np.add(mask[:,:,:,1], mask[:,:,:,2])

            # Clip miniscus and tibial masks at 1 in case the two overlap
            minisc_mask = np.clip(minisc_mask, 0, 1)
            tibial_mask = np.clip(tibial_mask, 0, 1)

            # Initialise final masks dimensions using existing masks, but switch 4 class to 6
            # Adjust mask dimension from 6 classes to 4 classes
            mask_dims = mask.shape[:-1]
            mask_dims += (self.num_classes,)
            mask_dims
            mask_all = np.zeros(mask_dims) # Initalise mask using previosuly defined dimensions


            # Fill in each layer of multiclass mask with each classes seg mask
            mask_all[:,:,:,1] = mask[:,:,:,0]
            mask_all[:,:,:,2] = tibial_mask 
            mask_all[:,:,:,3] = mask[:,:,:,2]
            mask_all[:,:,:,4] = minisc_mask
            
            # Define background mask for the train data

            # Identify positions where every other classes are zero 
            all_classes_zero_mask = np.all(mask_all == 0, axis=3)

            # Set background masks to be intergers (so background equal to 1 where all other sare zero)
            background_mask = all_classes_zero_mask.astype(int)
            
            # Set appropriate multiclass mask slice to backgrond mask
            mask_all[:,:,:,0] = background_mask

            # Change dimension order to match prediction output dimensions for loss function
            mask = mask_all.transpose(3,0,1,2)
        
        # Crop images and masks
        # image = crop_im(image, dim1_lower=24, dim1_upper=312, dim2_lower=26, dim2_upper=314)
        image = crop_im(image, 
                        dim1_lower=self.img_crop[0][0], 
                        dim1_upper=self.img_crop[0][1], 
                        dim2_lower=self.img_crop[1][0], 
                        dim2_upper=self.img_crop[1][1])
        
        # mask = crop_mask(mask, dim1_lower=24, dim1_upper=312, dim2_lower=26, dim2_upper=314)
        mask = crop_mask(mask, 
                         dim1_lower=self.img_crop[0][0], 
                         dim1_upper=self.img_crop[0][1], 
                         dim2_lower=self.img_crop[1][0], 
                         dim2_upper=self.img_crop[1][1])

        # Normalise image
        image = clip_and_norm(image, 0.005)

        # Create torch tensor from numpy, add channel dimension
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        # Apply transforms
        if self.transform != None:

            # Manually apply trasnforms with randomisation as image and mask must have the same transforms applied
            # Generate a random number, if above a threshold apply the transform to the image and the mask 
            if random.random() < self.transform_chance:
                image = self.transform(image)
                mask = self.transform(mask)

        return image, mask