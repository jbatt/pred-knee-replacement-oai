import os
import numpy as np
import nibabel as nib
from pathlib import Path
import SimpleITK as sitk
import glob
import pydicom
import shutil

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




    # TODO: Implement your data conversion logic here
    # 4. Update dataset.json with correct numbers

def generate_nnunet_dataset(raw_data_path, nnunet_raw_path, nnunet_dataset_name="Dataset014_OAISubset"):
    """
    Prepare dataset for nnU-Net by organizing files in the required structure.
    
    Args:
        raw_data_path (str): Path to your raw data
        nnunet_raw_path (str): Path to nnU-Net raw data directory
        task_name (str): Name of the task following nnU-Net conventions
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
    image_files = sorted(glob.glob(os.path.join(raw_data_path, "*.im")))
    label_files = sorted(glob.glob(os.path.join(raw_data_path, "*.seg")))
    print(f"Number of images to be converted: {len(image_files)}")
    print(f"Number of images to be converted: {len(label_files)}")


    # Loop through files, combine image and label files and convert to nii.gz format
    print("Converting images...")

    for i, (image_file, label_file) in enumerate(zip(image_files, label_files)):
        print(f"Converting image {i+1} of {len(image_files)}")
        # Define filename prefix
        dest_filename = f"OAI_{i:03d}_0000"
        # Define image and label filenames
        image_dest_filepath = os.path.join(imagesTr, dest_filename)
        label_dest_filepath = os.path.join(labelsTr, dest_filename)

        # Convert image and label files to nii.gz format
        # convert_to_nii_gz(input_file=image_file, output_file=image_dest_filepath)
        # convert_to_nii_gz(input_file=label_file, output_file=label_dest_filepath)
        shutil.copy(image_file, imagesTr)
        shutil.copy(label_file, labelsTr)

        
        # # Check if file is a training image
        # if "train_" in file and ".im" in file:
        #     # Set name of estination ile to align with nnU-Net conventions
        #     dest_filename = f"OAI_{i:03d}_0000"
        #     dest_filepath = os.path.join(imagesTr, dest_filename)
            
        #     # shutil.copy(os.path.join(raw_data_path, file), os.path.join(imagesTr, destination_filename))
        #     convert_to_nii_gz(input_file=file, output_file=dest_filepath)

        # if "train_" in file and ".seg" in file:
        #     # Set name of estination ile to align with nnU-Net conventions
        #     dest_filename = f"OAI_{i:03d}_0000"
        #     dest_filepath = os.path.join(labelsTr, dest_filename)

        #     # shutil.copy(os.path.join(raw_data_path, file), os.path.join(labelsTr, destination_filename))
        #     convert_to_nii_gz(input_file=file, output_file=dest_filepath)
        

        # TODO: convert test image as well

        # else:
        #     print(f"Unknown file: {file}")
    

    # Create dataset.json template
    # dataset_dict = {
    #     "name": "Knee Replacement",
    #     "description": "Knee replacement prediction dataset",
    #     "tensorImageSize": "3D",
    #     "reference": "",
    #     "licence": "",
    #     "release": "0.0",
    #     "modality": {
    #         "0": "CT",
    #     },
    #     "labels": {
    #         "0": "background",
    #         "1": "knee",
    #     },
    #     "numTraining": 0,  # Update this with actual number
    #     "numTest": 0,      # Update this with actual number
    # }


def main():
    # Set your paths here
    raw_data_path = "/mnt/scratch/scjb/data/oai_subset/train"
    nnunet_raw_path = "/mnt/scratch/scjb/nnUNet_raw"
    
    # Prepare the dataset
    generate_nnunet_dataset(raw_data_path, nnunet_raw_path)
    print("Done!")

if __name__ == "__main__":
    main()