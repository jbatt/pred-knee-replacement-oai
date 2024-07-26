import torch
import torch.nn as nn

# Define Dice coefficient for predicted and ground truth masks
def dice_coefficient(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    """Returns the Dice coefficient for between two masks

    Args:
        pred_mask (torch.Tensor): predicted mask
        gt_mask (torch.Tensor): ground truth mask

    Returns:
        float: determined Dice coefficient
    """

    # Sum the product of the two masks (e.g. the number of voxels they overlap)
    intersection = torch.sum(pred_mask * gt_mask)
    
    # Sum the total area of both masks 
    size = pred_mask.sum().item() + gt_mask.sum().item()
    
    # Calculate dice score as twice the intersection divided 
    dice = (2.0 * intersection) / size

    return dice.item()



# Return average dice coefficient for a batch of input and target masks
# 'smooth' constant included to avoid NaN errors when volume is zero
def dice_coefficient_batch(pred_mask_batch: torch.Tensor, gt_mask_batch: torch.Tensor, smooth=1e-5):
    """Returns the dice coefficient for a batch of predicted and ground truth masks.

    Args:
        pred_mask_batch (torch.Tensor): Predicted mask batch
        gt_mask_batch (torch.Tensor): Ground truth mask batch
        smooth (float, optional): Constant to avoid NaN errors when volume is zero. Defaults to 1e-5.

    Returns:
        float: Dice coefficient for input batch
    """
    # Start from third element (i.e. start of spatial dimensions)
    spatial_dims = tuple(range(2, len(pred_mask_batch.shape)))
    
    # Calculate the intersection as the sum over spatial dimensions the product of the two masks
    intersection = torch.sum(pred_mask_batch * gt_mask_batch, dim=spatial_dims)
    
    # Separately sum each mask over their respective spatial dimenions
    size = torch.sum(pred_mask_batch, dim=spatial_dims) + torch.sum(gt_mask_batch, dim=spatial_dims)
    
    # Calculate Dice score
    dice = (2.0 * intersection + smooth) / (size + smooth)

    # Return mean dice coeff of batch
    return torch.mean(dice)


# Caluclate Dice loss as 1 - dice coefficient
def dice_loss_batch(pred_mask_batch: torch.Tensor, gt_mask_batch: torch.Tensor, smooth=1e-5):
    """Returns the Dice loss calculated as 1 - Dice coefficient

    Args:
        pred_mask_batch (torch.Tensor): Predicted mask batch
        gt_mask_batch (torch.Tensor): Ground truth mask batch
        smooth (float, optional): Constant to avoid NaN errors when volume is zero. Defaults to 1e-5.

    Returns:
        float: Dice loss for the input batch
    """
    mean_loss = 1 - dice_coefficient_batch(pred_mask_batch, gt_mask_batch, smooth=smooth)

    return mean_loss


# Loss includes both binary cross-entropy and dice loss
# binary cross-entropy loss and dice loss are summed
def bce_dice_loss_batch(pred_mask_batch, gt_mask_batch):
    
    """Returns batch loss caluclated as the sum of the binary cross-entropy loss and dice loss.

    Args:
        pred_mask_batch (torch.Tensor): Predicted mask batch
        gt_mask_batch (torch.Tensor): Ground truth mask batch
        
    Returns:
        float: Binary cross-entropy and Dice loss summed
    """
    dice = dice_loss_batch(pred_mask_batch, gt_mask_batch)
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred_mask_batch, gt_mask_batch)
    
    return bce + dice