import torch
import torch.nn as nn
import sys
import gc
import numpy as np
import monai
from skimage import morphology
import scipy.ndimage as ndi 


# TODO: add ASSD metric loss function as hyperparameter
    # MONAI ASSD function
    # Add ASSD metric
    # Add ASSD as loss
    # Create loss hyperparameter in config file
    # Add in loss hyperparam in train.py


# Include src directory in path to import custom modules
if '..\\..\\src' not in sys.path:
    sys.path.append('..\\..\\src')

from utils.utils import debug_print


# Define Dice coefficient for predicted and ground truth masks
def dice_coefficient(pred_mask: torch.Tensor, gt_mask: torch.Tensor, debug=False) -> float:
    """Returns the Dice coefficient for between two masks

    Args:
        pred_mask (torch.Tensor): predicted mask
        gt_mask (torch.Tensor): ground truth mask

    Returns:
        float: determined Dice coefficient
    """

    # Sum the product of the two masks (e.g. the number of voxels they overlap)
    intersection = torch.sum(pred_mask * gt_mask)
    debug_print(debug, f"intersection = {intersection}")

    # Sum the total area of both masks 
    size = pred_mask.sum().item() + gt_mask.sum().item()
    debug_print(debug, f"size = {size}")

    # Calculate dice score as twice the intersection divided 
    dice = (2.0 * intersection) / size

    return dice.item()





# Return average dice coefficient for a batch of input and target masks
# 'smooth' constant included to avoid NaN errors when volume is zero
def dice_coefficient_batch(pred_mask_batch: torch.Tensor, 
                           gt_mask_batch: torch.Tensor, 
                           smooth=1e-5) -> torch.Tensor:
    
    """Returns the dice coefficient for a batch of predicted and 
    ground truth masks.

    Args:
        pred_mask_batch (torch.Tensor): Predicted mask batch
        gt_mask_batch (torch.Tensor): Ground truth mask batch
        smooth (float, optional): Constant to avoid NaN errors 
        when volume is zero. Defaults to 1e-5.

    Returns:
        float: Dice coefficient for input batch
    """
    # Release all unoccupied cached memory
    gc.collect()
    torch.cuda.empty_cache()

    # Start from third element (i.e. start of spatial dimensions)
    print(f"Dice single pred shape: {pred_mask_batch.shape}")
    print(f"Dice single mask shape: {gt_mask_batch.shape}")
    
    # spatial_dims = tuple(range(1, len(pred_mask_batch.shape)))
    # print(f"spatial_dims = {spatial_dims}")

    # Calculate the intersection as the sum over spatial dimensions the product of the two masks
    intersection = torch.sum(pred_mask_batch * gt_mask_batch)

#    intersection = torch.sum(pred_mask_batch * gt_mask_batch, dim=spatial_dims)
    print(f"intersection = {intersection}")


    # Separately sum each mask over their respective spatial dimenions
#    size = torch.sum(pred_mask_batch, dim=spatial_dims) + torch.sum(gt_mask_batch, dim=spatial_dims)
    size = torch.sum(pred_mask_batch) + torch.sum(gt_mask_batch)

    print(f"size = {size}")

    # Calculate Dice score
    dice = (2.0 * intersection + smooth) / (size + smooth)

    # Return mean dice coeff of batch
    return torch.mean(dice)



def dice_coefficient_multi_batch_all(pred_mask_batch, gt_mask_batch, smooth=1e-5):
    
    """Calculates multi-class Dice coefficient 
    
    Takes input multi-class predicted masks (logit values) and ground truth masks (binary values) and outputs a tensor with a dice score for each class

    Args:
        pred_mask_batch (torch.Tensor): Predicted multi-class **logit** mask - expected dims (num_classes, height, width, depth): 
        gt_mask_batch (torch.Tensor): Ground truth multi-class **binary** mask batch - expected dims (batch, num_classes, height, width, depth)
        smooth (float, optional): Constant to avoid NaN errors 
        when volume is zero. Defaults to 1e-5.
    
    Returns:
        tensor: Dice score for each class 
    """


    # Release all unoccupied cached memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Ensure pred_mask_batch is probailities using softmax and not logits 
    pred_mask_batch = nn.functional.softmax(pred_mask_batch, dim=1)
    
    print(f"Multiclass Dice loss pred_mask_batch shape = {pred_mask_batch.shape}")
    print(f"Multiclass Dice loss gt_mask_batch shape = {gt_mask_batch.shape}")
    
    # Remove dim of 1 from predicted mask and ground truth 
    gt_mask_batch = torch.squeeze(gt_mask_batch, dim=1)


    # Flatten the tensors to shape (batch_size, num_classes, num_elements)
    y_true_flat = gt_mask_batch.view(gt_mask_batch.size(0), gt_mask_batch.size(1), -1)
    y_pred_flat = pred_mask_batch.view(pred_mask_batch.size(0), pred_mask_batch.size(1), -1)

    # Calculate intersection and union for each class
    intersection = (y_true_flat * y_pred_flat).sum(dim=2)
    size = y_true_flat.sum(dim=2) + y_pred_flat.sum(dim=2)

    # Calculate Dice coefficient for each class
    dice_per_class = (2. * intersection + smooth) / (size + smooth)
    
    # Take mean over batch dimension
    dice_per_class = dice_per_class.mean(dim=0)

    return dice_per_class



def dice_coefficient_multi_batch(pred_mask_batch, gt_mask_batch, smooth=1e-5):

    """Calculates the mean multi-class Dice coefficient (by class) 
    
    Takes input multi-class predicted masks (logit values) and ground truth masks (binary values) and outputs a tensor with a dice score for each class

    Args:
        pred_mask_batch (torch.Tensor): Predicted multi-class **logit** mask - expected dims (num_classes, height, width, depth): 
        gt_mask_batch (torch.Tensor): Ground truth multi-class **binary** mask batch - expected dims (batch, num_classes, height, width, depth)
        smooth (float, optional): Constant to avoid NaN errors 
        when volume is zero. Defaults to 1e-5.
    
    Returns:
        tensor: Mean Dice score - average taken across each class 
    """

    # Release all unoccupied cached memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Ensure pred_mask_batch is probailities using softmax and not logits 
    pred_mask_batch = nn.functional.softmax(pred_mask_batch, dim=1)
    
    print(f"Multiclass Dice loss pred_mask_batch shape = {pred_mask_batch.shape}")
    print(f"Multiclass Dice loss gt_mask_batch shape = {gt_mask_batch.shape}")
    
    # Remove dim of 1 from predicted mask and ground truth 
    gt_mask_batch = torch.squeeze(gt_mask_batch, dim=1)

    # Flatten the tensors to shape (batch_size, num_classes, num_elements)
    y_true_flat = gt_mask_batch.view(gt_mask_batch.size(0), gt_mask_batch.size(1), -1)
    y_pred_flat = pred_mask_batch.view(pred_mask_batch.size(0), pred_mask_batch.size(1), -1)

    # Calculate intersection and union for each class
    intersection = (y_true_flat * y_pred_flat).sum(dim=2)
    size = y_true_flat.sum(dim=2) + y_pred_flat.sum(dim=2)

    # Calculate Dice coefficient for each class
    dice_per_class = (2. * intersection + smooth) / (size + smooth)

    # Return the mean Dice coefficient across all classes
    mean_dice = dice_per_class.mean()

    print(f"Mean dice = {mean_dice}")

    return mean_dice




#########################################################################
# ASSD
#########################################################################

def average_surface_distance(pred_mask, gt_mask, spacing=[0.36,0.36,0.7]):

    # ASSD function expects batch dimension so add dim back
    pred_mask = np.expand_dims(pred_mask, axis=0)
    gt_mask = np.expand_dims(gt_mask, axis=0)

    # Compute ASSD - [0.36,0.36,0.7] is voxel resolution
    assd = monai.metrics.compute_average_surface_distance(pred_mask, gt_mask, symmetric=True, include_background=True, spacing=spacing)
    
    return assd



#########################################################################
# THICKNESS
#########################################################################

def calculate_thickness(segmentation):
    """Calculate the thickness of a 3D segmented region in an input mask
    using the medial axis and distance transform of the segmentation mask"
    
    Args:
        segmentation (np.array): a 2D, 3D or 4D (onehot encoded) array representing a segmented region
    """

    # If a 2 or 3 dimensional array calculate the distance transform and skeletonise using the standard methods
    if len(segmentation.shape) in (2,3):

        distance_transform = ndi.distance_transform_edt(segmentation)
        segmentation_skeleton = morphology.skeletonize(segmentation, method="lee")
        
    # If a 4-dimensional array is passed, loop through the first dimension (e.g. class for onehot encoded mask)
    if len(segmentation.shape) == 4:

        print("Input sgementation mask is 4D - looping thorugh onehot encoded masks...")
        
        # Calculate the distance transform of the mask
        distance_transform = np.zeros(segmentation.shape).astype(np.uint8)
        for i in range(segmentation.shape[0]):
            distance_transform[i,...] = ndi.distance_transform_edt(segmentation[i,...]).astype(np.uint8)
            # print(f"Unique values in distance transform: {np.unique(distance_transform)}")
        

        # Calculate the skeletonised distance transform
        segmentation_skeleton = np.zeros(segmentation.shape, dtype=np.uint8)
        
        for i in range(segmentation.shape[0]):    
            current_volume = segmentation[i,...]
            # print(f"Current volume shape: {current_volume.shape}")
            # print(f"Current volume unqiue values: {np.unique(current_volume)}")

            segmentation_skeleton[i,...] = morphology.skeletonize(current_volume).astype(np.uint8)
        
                    
        print(f"Segmentation skeleton shape: {segmentation_skeleton.shape}")
        print(f"Segmentation skeleton values: {np.unique(segmentation_skeleton)}")

    # Determine thickness as the distance transform of the skeletonised mask multiplied by 2
    thickness = distance_transform * segmentation_skeleton * 2
    print(f"Thickness shape: {thickness.shape}")
    #print(f"Thickness\n: {thickness.astype(np.uint8)}")

    # Extract non-zero values
    print(f"Thickness values: {np.unique(thickness)}")
    
    shape_thickness_values = []

    for i in range (segmentation.shape[0]): 
        non_zero_thickness = list(thickness[i, ...][thickness[i, ...] != 0])
        # print(non_zero_thickness)
        shape_thickness_values.append(non_zero_thickness)

    # print(shape_thickness_values)
    mean_thickness = [np.mean(x) for x in shape_thickness_values]

    return mean_thickness