import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
import os
import numpy as np
from utils.utils import crop_im, crop_mask, clip_and_norm
from monai.data import GridPatchDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    RandCropByLabelClassesd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
)


# TODO - implement stride in patch creation 

# Multiclass knee cartilage MRI volumes and segmentation masks
class KneeSegDataset3DMulticlass(Dataset):
    def __init__(self, 
                 file_paths, 
                 data_dir, 
                 num_classes, 
                 img_crop=[[25,281],[25,281],[0,160]], 
                 split='train', 
                 transform=None, 
                 transform_chance=0.5, 
                 patch_size=None,
                 patch_stride=None,
                 patch_method = None,
                 num_patches=1):
        
        self.file_paths = file_paths
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.transform_chance = transform_chance
        self.num_classes = num_classes
        self.img_crop = img_crop
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_method = patch_method
        self.num_patches = num_patches
        


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
            # Take all orignal mask dimensions
            mask_dims = mask.shape[:-1]
            # Add a new dimension for the number of classes
            mask_dims += (self.num_classes,)
            print(f"mask dims: {mask_dims}")
            mask_all = np.zeros(mask_dims) # Initalise mask using previosuly defined dimensions


            # Fill in each layer of multiclass mask with each classes seg mask
            mask_all[:,:,:,1] = mask[:,:,:,0] # Femoral cartilage
            mask_all[:,:,:,2] = tibial_mask # Tibial cartilage
            mask_all[:,:,:,3] = mask[:,:,:,3] # Patellar cartilage
            mask_all[:,:,:,4] = minisc_mask # Meniscus
            
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



        # Extract patches using sliding widnow over full volume based on patch_size and patch_stride (using Pytorch torch.unfold) 
        if self.patch_method == "grid_pytorch" and self.patch_size is not None:
            # Original image and mask shapes
            print(f"Original image shape: {image.shape}")
            print(f"Original mask shape {mask.shape}")

            # Unfold image into patches
            image = image.unfold(1,self.patch_size[0], self.patch_stride[0]).unfold(2,self.patch_size[1], self.patch_stride[1]).unfold(3,self.patch_size[2], self.patch_stride[2])
            print(f"Unfolded image shape: {image.shape}")

            # Reshape image to have patches as the first dimension
            image = image.reshape(-1, self.patch_size[0], self.patch_size[1], self.patch_size[2])
            print(f"Reshaped image shape: {image.shape}")
            
            # Unfold mask into patches
            mask = mask.unfold(2, self.patch_size[0], self.patch_stride[0]).unfold(3, self.patch_size[1], self.patch_stride[1]).unfold(4, self.patch_size[2], self.patch_stride[2])
            print(f"Unfolded mask shape: {mask.shape}")

            mask = mask.reshape(-1, self.num_classes, self.patch_size[0], self.patch_size[1], self.patch_size[2])
            print(f"Reshaped mask shape: {mask.shape}")            



        # Apply monai patch extraction (random patches)   
        if self.patch_method == "random_monai" and self.patch_size is not None:
            

            print(f"Original image shape: {image.shape}")
            print(f"Original mask shape {mask.shape}")

            # # Add channel dimension of 1 to image to match MONAI requirements
            # image = image.unsqueeze(0) 

            mask = mask.squeeze(0)  # Remove the channel dimension from the mask
            
            # Convert mask to single channel from onehot encoding
            # mask = torch.argmax(mask, dim=0, keepdim=True)

            print(f"Image shape after unsqueeze: {image.shape}")

            print(f"Extracting patches using MOANI random patch extraction...")
            data_dict = {
                "image": image,
                "label": mask,
            }
            
            # Define patch extraction and augmentation transforms
            self.patch_transform = Compose([

                # Define patch extraction transform
                RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=self.patch_size,
                ratios=[1, 1, 1, 1, 1], # centre patches on each class with equal probability
                num_classes=self.num_classes,
                num_samples=self.num_patches,  # Just one patch per call TODO: update this
                ),
                # Patch data augmentation
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                ToTensord(keys=["image", "label"]),
            ])


            # Apply MONAI patch extraction
            data_dict = self.patch_transform(data_dict)
            # Optional: Convert back to tuple format if needed
            print(f"Data dict 0 after MONAI patch extraction: {data_dict[0]["image"].shape}")
            print(f"Data dict 1 after MONAI patch extraction: {data_dict[0]["label"].shape}")

            image = data_dict[0]["image"]
            mask = data_dict[0]["label"]

            print(f"Image shape after MONAI patch extraction: {image.shape}")
            print(f"Mask shape after MONAI patch extraction: {mask.shape}")



        # Apply transforms
        if self.transform != None:

            # Manually apply trasnforms with randomisation as image and mask must have the same transforms applied
            # Generate a random number, if above a threshold apply the transform to the image and the mask 
            if random.random() < self.transform_chance:
                image = self.transform(image)
                mask = self.transform(mask)


        return image, mask
    




class LengthEnabledGridPatchDataset(GridPatchDataset):
    def __init__(self, data, patch_iter):
        super().__init__(data=data, patch_iter=patch_iter)
        self.total_patches = self._count_total_patches(patch_iter)
    
    def _count_total_patches(self, patch_iter):
        print(f"patch_iter: {patch_iter.patch_size}")
        print(f"data shape: {len(self.data)}")

        len_data = len(self.data)

        sample = self.data[0]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample
        print(f"image shape: {image.shape}")
        
        num_patches_per_image = (
                 (image.shape[1] * image.shape[2] * image.shape[3]) /
                 (patch_iter.patch_size[1] * patch_iter.patch_size[2] * patch_iter.patch_size[3])
        )
        print(f"num_patches_per_image: {num_patches_per_image}")

        total_patches = len_data * num_patches_per_image


        # for i in range(len(self.data)):
        #     # Get sample from the underlying dataset
        #     sample = self.data[i]
            
        #     # Extract image shape (assuming image is the first element in sample)
        #     # Adjust this based on your actual data structure
        #     image = sample[0] if isinstance(sample, (tuple, list)) else sample

        #     print(f"image shape: {image.shape}")
            
        #     # Count patches for this sample
        #     num_patches = (
        #         (image.shape[1] * image.shape[2] * image.shape[3]) /
        #         (patch_iter.patch_size[1] * patch_iter.patch_size[2] * patch_iter.patch_size[3])
        #     )
        #     total += num_patches
            
        print(f"total patches: {total_patches}")
        return int(total_patches)
    
    def __len__(self):
        return self.total_patches