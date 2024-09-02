import torch
import torch.nn as nn
import sys
import torch.nn as nn
import gc


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




def dice_coefficient_multi_batch(pred_mask_batch, gt_mask_batch, num_labels, smooth=1e-5):
    
    # Release all unoccupied cached memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Ensure pred_mask_batch is probailities using softmax and not logits 
    pred_mask_batch = nn.functional.softmax(pred_mask_batch, dim=1)
    
    print(f"Multiclass Dice loss pred_mask_batch shape = {pred_mask_batch.shape}")
    print(f"Multiclass Dice loss gt_mask_batch shape = {gt_mask_batch.shape}")
    
    # Remove dim of 1 from predicted mask and ground truth 
    gt_mask_batch = torch.squeeze(gt_mask_batch, dim=1)
    
    # dice = 0
    # for index in range(num_labels):
        
    #     # Release all unoccupied cached memory
    #     gc.collect()
    #     torch.cuda.empty_cache()
         
    #     dice += float(dice_coefficient_batch(pred_mask_batch[:,index,:,:,:], gt_mask_batch[:,index,:,:,:], smooth=smooth))
    
    # print(f"\nDice score = {dice}\n")
    # return dice / num_labels # Returnn average dice from all class labels


    # Flatten the tensors to shape (batch_size, num_classes, num_elements)
    y_true_flat = gt_mask_batch.view(gt_mask_batch.size(0), gt_mask_batch.size(1), -1)
    y_pred_flat = pred_mask_batch.view(pred_mask_batch.size(0), pred_mask_batch.size(1), -1)

    # Calculate intersection and union for each class
    intersection = (y_true_flat * y_pred_flat).sum(dim=2)
    union = y_true_flat.sum(dim=2) + y_pred_flat.sum(dim=2)

    # Calculate Dice coefficient for each class
    dice_per_class = (2. * intersection + smooth) / (union + smooth)

    # Return the mean Dice coefficient across all classes
    mean_dice = dice_per_class.mean()

    print(f"Mean dice = {mean_dice}")
    
    return mean_dice




# Caluclate Dice loss as 1 - dice coefficient
def dice_loss_batch(pred_mask_batch: torch.Tensor, 
                    gt_mask_batch: torch.Tensor,
                    num_labels, 
                    smooth=1e-5
                    ):
    """Returns the Dice loss calculated as 1 - Dice coefficient

    Args:
        pred_mask_batch (torch.Tensor): Predicted mask batch
        gt_mask_batch (torch.Tensor): Ground truth mask batch
        smooth (float, optional): Constant to avoid NaN errors 
        when volume is zero. Defaults to 1e-5.

    Returns:
        float: Dice loss for the input batch
    """
    mean_loss = 1 - dice_coefficient_batch(pred_mask_batch, 
                                           gt_mask_batch, 
                                           num_labels)

    return mean_loss


# Caluclate Dice loss as 1 - dice coefficient
def dice_loss_multi_batch(pred_mask_batch: torch.Tensor, 
                    gt_mask_batch: torch.Tensor,
                    num_labels, 
                    smooth=1e-5
                    ):
    """Returns the Dice loss calculated as 1 - Dice coefficient

    Args:
        pred_mask_batch (torch.Tensor): Predicted mask batch
        gt_mask_batch (torch.Tensor): Ground truth mask batch
        smooth (float, optional): Constant to avoid NaN errors 
        when volume is zero. Defaults to 1e-5.

    Returns:
        float: Dice loss for the input batch
    """
    mean_loss = 1 - dice_coefficient_multi_batch(pred_mask_batch, 
                                           gt_mask_batch, 
                                           num_labels)

    return mean_loss


# Loss includes both binary cross-entropy and dice loss (summed)
def bce_dice_loss_batch(pred_mask_batch, gt_mask_batch):
    
    """Returns batch loss caluclated as the sum of the binary 
    cross-entropy loss and dice loss.

    Args:
        pred_mask_batch (torch.Tensor): Predicted mask batch
        gt_mask_batch (torch.Tensor): Ground truth mask batch
        
    Returns:
        float: Binary cross-entropy and Dice loss summed
    """

    # Caluclate dice loss for batch
    dice = dice_loss_multi_batch(pred_mask_batch, gt_mask_batch)
    
    # Caluclate binary-cross entropy loss
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred_mask_batch, gt_mask_batch)
    
    return bce + dice



# Loss includes both cross-entropy and dice loss (summed)
def ce_dice_loss_multi_batch(pred_mask_batch_logits, gt_mask_batch, num_labels):
    
    """Returns batch loss caluclated as the sum of the  
    cross-entropy loss and dice loss.

    Args:
        pred_mask_batch (torch.Tensor): Predicted mask batch
        gt_mask_batch (torch.Tensor): Ground truth mask batch
        
    Returns:
        float: Binary cross-entropy and Dice loss summed
    """

    # Release all unoccupied cached memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Remove dim of 1 from predicted mask and ground truth 
    gt_mask_batch = torch.squeeze(gt_mask_batch, dim=1)
    print(f"squeezed gt_mask_batch shape = {gt_mask_batch.shape}")
    print(f"pred_mask_batch_logits shape = {pred_mask_batch_logits.shape}")
    
    # Calculate softmax to generate probabiltiies for dice score
    pred_mask_batch_probs = nn.functional.softmax(pred_mask_batch_logits, dim=1)

    # Caluclate dice loss for batch using probabilities
    dice = dice_loss_multi_batch(pred_mask_batch_probs, gt_mask_batch, num_labels)
    
    # Caluclate cross entropy loss using logits
    ce_loss = nn.CrossEntropyLoss()
    ce = ce_loss(pred_mask_batch_logits, gt_mask_batch)
    
    return ce + dice