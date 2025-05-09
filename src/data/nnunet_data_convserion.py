import os
import numpy as np
import nibabel as nib
from pathlib import Path
import SimpleITK as sitk
import glob
import pydicom
import shutil
import h5py
import sys
import json
import argparse

# Include src directory in path to import custom modules
if '../src' not in sys.path:
    sys.path.append('../src')

print(f"sys.path = {sys.path}")



from utils.utils import crop_im, crop_mask, clip_and_norm

def convert_to_nifti(input_file, output_file=None):
    try:
        data = pydicom.dcmread(input_file, force=True)

        # Extract pixel data
        pixel_data = data.pixel_array

        # Create NIfTI image
        nifti_image = nib.Nifti1Image(pixel_data, np.eye(4))

        if output_file is None:
            output_file = os.path.splitext(input_file)[0] + ".nii.gz"
        
        # Save NIfTI image
        nib.save(nifti_image, output_file)

        return output_file
    
    except Exception as e:
        print(f"Error converting file {input_file}: {e}")
        return None




def convert_to_nii_gz(input_file, output_file=None):
    """
    Convert a file to nii.gz format.
    
    Args:
        input_file (str): Path to the input file
        output_file (str): Path to the output file. If None, output file will be saved in the same directory as input file.
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".nii.gz"
    
    # Read the image
    image = sitk.ReadImage(input_file)
    
    # Write the image
    sitk.WriteImage(image, output_file, useCompression=True)
    
    return output_file



def save_numpy_to_nifti(numpy_array: np.array, output_filepath, affine=None):
    """
    Save a numpy array as a NIfTI file

    Args:
        numpy_array: Numpy array containing image data
        output_file: Path where to save the NIfTI (.nii or .nii.gz)
        affine (optional): defaults to identity if None
    """

    if affine is None:
        affine = np.eye(4)

    nifti_image = nib.Nifti1Image(numpy_array, affine)

    nib.save(nifti_image, output_filepath)

    return output_filepath



    # TODO: Implement your data conversion logic here
    # 4. Update dataset.json with correct numbers

def generate_nnunet_dataset(data_path, 
                            nnunet_raw_path, 
                            nnunet_dataset_name="Dataset014_OAISubset", 
                            train=False, 
                            test=False,
                            img_crop=[[25,281], [25,281], [0,160]]) -> None:
    """
    Prepare dataset for nnU-Net by organizing files in the required structure.
    
    Args:
        raw_data_path (str): Path to your raw data
        nnunet_raw_path (str): Path to nnU-Net raw data directory
        task_name (str): Name of the task following nnU-Net conventions
        train (int): Whether to convert training data (1) or not (0)
        test (int): Whether to convert test data (1) or not (0)
        test_data_path (str): Path to your test data
    """
    # Create task directory
    task_dir = os.path.join(nnunet_raw_path, nnunet_dataset_name)

    # Set nnUNet data directories for train, labels and test
    imagesTr = os.path.join(task_dir, "imagesTr")
    labelsTr = os.path.join(task_dir, "labelsTr")
    imagesTs = os.path.join(task_dir, "imagesTs")
    
    # Create directories if they don't exist
    print("Making directories...")
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(imagesTs, exist_ok=True)

    # List all files in raw data directory and sort them
    #files = sorted(os.listdir(raw_data_path))

    # List image and label files and sort them
    train_data_path = os.path.join(data_path, "train")
    test_data_path = os.path.join(data_path, "test")

    train_image_files = sorted(glob.glob(os.path.join(train_data_path, "*.im")))
    label_files = sorted(glob.glob(os.path.join(train_data_path, "*.seg")))


    # print(f"Image files: {image_files}")
    # print(f"Label files: {label_files}")

    # Loop through files, combine image and label files and convert to nii.gz format
    print("Converting images...")


    #########################################################################
    # CONVERT TRAINING DATA
    #########################################################################
    if train:

        for i, (train_image_filepath, label_filepath) in enumerate(zip(train_image_files, label_files)):
            print(f"Converting image {i+1} of {len(train_image_filepath)}")
            print(f"Image file being converted: {train_image_filepath}")
            
            # Define filename prefix
            image_dest_filename = f"OAI_{i:03d}_0000.nii.gz"
            label_dest_filename = f"OAI_{i:03d}.nii.gz"
            

            # Define image and label filenames
            image_dest_filepath = os.path.join(imagesTr, image_dest_filename)
            label_dest_filepath = os.path.join(labelsTr, label_dest_filename)
            


            # Open the image file and load it to a numpy array
            if train:
                with h5py.File(train_image_filepath,'r') as hf:
                    image = np.array(hf['data'])
                    print(f"Image shape: {image.shape}")
                # Open the mask file and load it to a numpy array
                with h5py.File(label_filepath,'r') as hf:
                    mask = np.array(hf['data'])
                    print(f"Mask shape: {mask.shape}")

                # Define medial meniscus mask
                menisc_med_mask = mask[...,-1]

                # Define lateral meniscus mask
                # Below captures single train example with lateral meniscus mask at wrong index
                if os.path.splitext(image_dest_filepath)[0] == 'train_026_V01':
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
                num_classes = 5
                mask_dims += (num_classes,)
                mask_dims
                mask_all = np.zeros(mask_dims) # Initalise mask using previosuly defined dimensions


                # Fill in each layer of multiclass mask with each classes seg mask
                mask_all[:,:,:,1] = mask[:,:,:,0] # Femoral 
                mask_all[:,:,:,2] = tibial_mask # Tibial
                mask_all[:,:,:,3] = mask[:,:,:,3] # Patellar
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


                # Convert mask from one hot encoding to single channel and convert to uint8 type
                mask = np.argmax(mask, axis=0).astype(np.uint8)

                # Crop images and masks
                image = crop_im(image, dim1_lower=img_crop[0][0], dim1_upper=img_crop[0][1], dim2_lower=img_crop[1][0], dim2_upper=img_crop[1][1])
                mask = crop_mask(mask, dim1_lower=img_crop[0][0], dim1_upper=img_crop[0][1], dim2_lower=img_crop[1][0], dim2_upper=img_crop[1][1], onehot=False)

                # Normalise image
                image = clip_and_norm(image, 0.005)

                save_numpy_to_nifti(image, image_dest_filepath)
                save_numpy_to_nifti(mask, label_dest_filepath)

                print(f"{train_image_filepath} converted to nifti to location {image_dest_filepath}")
                print(f"{label_filepath} converted to nifti to location {label_dest_filepath}\n\n")



        #########################################################################
        # CONVERT TESTING DATA
        #########################################################################
    if test:
        print("Converting test images...")

        test_image_files = sorted(glob.glob(os.path.join(test_data_path, "*.im")))

        print(f"Test images: {test_image_files}")

        for i, test_image_filepath in enumerate(test_image_files):
            
            print(f"Converting image {i+1} of {len(test_image_files)}")
            print(f"Image file being converted: {test_image_filepath}")
            
            test_dest_filename = f"OAI_{i:03d}_0000.nii.gz"
            test_dest_filename = os.path.join(imagesTs, test_dest_filename)
            
            with h5py.File(test_image_filepath,'r') as hf:
                image = np.array(hf['data'])
                print(f"Image shape: {image.shape}")

            # Crop images
            image = crop_im(image, dim1_lower=img_crop[0][0], dim1_upper=img_crop[0][1], dim2_lower=img_crop[1][0], dim2_upper=img_crop[1][1])
            image = clip_and_norm(image, 0.005)

            save_numpy_to_nifti(image, test_dest_filename)

            print(f"{test_image_filepath} converted to nifti to location {test_dest_filename}")




def main():

    parser = argparse.ArgumentParser(description='Convert OAI dataset to nnU-Net format')
    parser.add_argument('--generate-data', type=int, help='Whether or not to generate the dataset')
    parser.add_argument('--generate-json', type=int, help='Whether or not to generate the dataset.json file')
    parser.add_argument('--train', help='Whether or not to convert training data', action=argparse.BooleanOptionalAction) # Default is False
    parser.add_argument('--test', help='Whether or not to convert test data', action=argparse.BooleanOptionalAction) # Default is False
    parser.add_argument('--img-crop', type=list, help='Image crop dimensions', default=[[25,281], [25,281], [0,160]])  
    
    args = parser.parse_args()

    print(f"Args: {args}")

    # Set your paths here
    raw_data_path = "/mnt/scratch/scjb/data/oai_subset/"
    nnunet_raw_path = "/mnt/scratch/scjb/nnUNet_raw"
    dataset_id = "Dataset014_OAISubset"

    # Prepare the dataset
    if args.generate_data == 1:
        generate_nnunet_dataset(raw_data_path, nnunet_raw_path, nnunet_dataset_name=dataset_id, train=args.train, test=args.test)


    if args.generate_json == 1:
        dataset_json = { 
            "channel_names": {
                "0": "DESS", 
            }, 
            "labels": {  
                "background": 0,
                "FC": 1,
                "TC": 2,
                "PC": 3,
                "M": 4
            }, 
            "numTraining": 120, 
            "file_ending": ".nii.gz"
        }

        with open(os.path.join(nnunet_raw_path, dataset_id, "dataset.json"), "w") as f:
            json.dump(dataset_json, f)


    print("Done!")

if __name__ == "__main__":
    main()