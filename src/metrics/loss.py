import torch
import torch.nn as nn
import sys
import gc
from metrics.metrics import dice_coefficient_batch, dice_coefficient_multi_batch, dice_coefficient_multi_batch_all
from monai.losses.hausdorff_loss import HausdorffDTLoss



# Caluclate Dice loss as 1 - dice coefficient
def dice_loss_batch(pred_mask_batch: torch.Tensor, 
                    gt_mask_batch: torch.Tensor,
                    num_labels, 
                    ):
    """Returns the Dice loss calculated as 1 - Dice coefficient. This uses the Dice score for a single-class segmentation mask.

    Args:
        pred_mask_batch (torch.Tensor): Predicted mask batch
        gt_mask_batch (torch.Tensor): Ground truth mask batch
        smooth (float, optional): Constant to avoid NaN errors 
        when volume is zero. Defaults to 1e-5.

    Returns:
        float: Dice loss for the input batch
    """
    mean_loss = 1 - dice_coefficient_batch(pred_mask_batch, 
                                           gt_mask_batch)

    return mean_loss


# Caluclate Dice loss as 1 - dice coefficient
def dice_loss_multi_batch(pred_mask_batch: torch.Tensor, 
                    gt_mask_batch: torch.Tensor, 
                    smooth=1e-5
                    ):
    """Returns the Dice loss calculated as 1 - Dice coefficient for multi-class segmentation masks

    Args:
        pred_mask_batch (torch.Tensor): Predicted mask batch (multi-class)
        gt_mask_batch (torch.Tensor): Ground truth mask batch (multi-class)
        smooth (float, optional): Constant to avoid NaN errors 
        when volume is zero. Defaults to 1e-5.

    Returns:
        float: Dice loss for the input batch
    """
    mean_loss = 1 - dice_coefficient_multi_batch(pred_mask_batch, 
                                           gt_mask_batch, 
                                           smooth)

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
def ce_dice_loss_multi_batch(pred_mask_batch_logits, gt_mask_batch):
    
    """Returns batch loss caluclated as the sum of the  
    cross-entropy loss and dice loss.

    Logit values are used for the predicted segmentation mask as the Pytorch cross-entropy function expects logit values. 

    Args:
        pred_mask_batch_logits (torch.Tensor): Predicted multi-class mask batch (logit values)
        gt_mask_batch (torch.Tensor): Ground truth mask batch (binary values)
        
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
    
    # Removed the softmax call below for now - think this leads to calling softmax twice on logits 

    # # Calculate softmax to generate probabiltiies for dice score
    # pred_mask_batch_probs = nn.functional.softmax(pred_mask_batch_logits, dim=1)

    # Changed to call on logits and softmax gets call in the Dice score calculation

    # Caluclate dice loss for batch using probabilities
    # dice = dice_loss_multi_batch(pred_mask_batch_probs, gt_mask_batch, num_labels)
    dice = dice_loss_multi_batch(pred_mask_batch_logits, gt_mask_batch)

    
    # Caluclate cross entropy loss using logits
    ce_loss = nn.CrossEntropyLoss()
    ce = ce_loss(pred_mask_batch_logits, gt_mask_batch)
    
    return ce + dice


def hausdorff_loss(pred_mask_batch, gt_mask_batch):
    
    """Returns batch Hausdorff loss.

    Args:
        pred_mask_batch_logits (torch.Tensor): Predicted multi-class mask batch (logit values)
        gt_mask_batch (torch.Tensor): Ground truth mask batch (binary values)
        
    Returns:
        float: Cross-Entropy, Dice loss and Hausdorff Loss summed
    """
    pred_mask_batch = torch.squeeze(pred_mask_batch, dim=1)
    hausdorff_loss_fn = HausdorffDTLoss(softmax=True, reduction="none", include_background=True)

    hausdorff_loss = hausdorff_loss_fn(pred_mask_batch, gt_mask_batch)
    
    return hausdorff_loss




def ce_dice_hausdorff_loss(pred_mask_batch, gt_mask_batch):

    """Returns batch loss caluclated as the sum of the  
    cross-entropy loss, dice loss and Hausdorff loss.

    Logit values are used for the predicted segmentation mask as the Pytorch cross-entropy function expects logit values. 

    Args:
        pred_mask_batch_logits (torch.Tensor): Predicted multi-class mask batch (logit values)
        gt_mask_batch (torch.Tensor): Ground truth mask batch (binary values)
        
    Returns:
        float: Cross-Entropy, Dice loss and Hausdorff Loss summed
    """
    ce_dice = ce_dice_loss_multi_batch(pred_mask_batch, gt_mask_batch)

    hausdorff = hausdorff_loss(pred_mask_batch, gt_mask_batch)

    return ce_dice + hausdorff




def create_loss(input_loss_arg):

    if input_loss_arg == "ce_dice":
        # Create CE, Dice loss
        loss = ce_dice_loss_multi_batch

    elif input_loss_arg == "ce_dice_hausdorff":
        # Create CE, Dice, Hausdorff
        loss = ce_dice_hausdorff_loss
    return loss