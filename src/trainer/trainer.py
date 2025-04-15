# Define functions to help with model training
import gc
import torch
import numpy as np

# from src.metrics.evaluation import dice_coefficient_multi_batch, dice_coefficient_multi_batch_all
from metrics.metrics import dice_coefficient_multi_batch, dice_coefficient_multi_batch_all

from monai.losses.hausdorff_loss import HausdorffDTLoss
from monai.metrics import compute_hausdorff_distance
from monai.inferers import sliding_window_inference

# Define a training loop function for reuse later 
def train_loop(
        dataloader, 
        device, 
        model, 
        loss_fn, 
        optimizer,
        scaler, 
        num_classes,
    ):

    print("Running training loop...")

    # Release all unoccupied cached memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load model to device
    print(f"Loading model to device: {device}")
    model.to(device)

    # Initialise variables
    epoch_loss = []
    epoch_dice = []
    epoch_haus = []

    # Initialise separate dice/hausdorff array which will capture dice of all tissues indvidually
    epoch_dice_all = np.empty(shape=(len(dataloader), num_classes))
    epoch_haus_loss_all = np.empty(shape=(len(dataloader), num_classes))


    size = len(dataloader.dataset)
    print(f"Dataset size = {size}")


    # Set the model to training mode
    print(f"Setting model to train mode...")
    model.train()

    print("Starting training loop...")
    
    # For each batch from the data loader
    for batch, (X, y) in enumerate(dataloader):

        # Release all unoccupied cached memory
        gc.collect()
        torch.cuda.empty_cache()

        X = X.to(device)
        y = y.to(device)
        
        print(f"Image shape: {X.size()}")
        print(f"Mask shape: {y.size()}")

        # Compute prediction and loss
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            print("Computing model predictions...")
            pred = model(X)
            assert pred.dtype is torch.float16
            print(f"Prediction shape: {pred.shape}")

            print("Calculating loss...")
            loss = loss_fn(pred, y)
            assert loss.dtype is torch.float32

        # Reset gradients to zero
        optimizer.zero_grad(set_to_none=True)

        # Backpropagation
        print("Computing backpropagation...")        
        scaler.scale(loss).backward()

        # Perform model step
        scaler.step(optimizer)
        scaler.update()


        
        # Store batch size
        batch_size = len(y)

        # # For every 5th batch, print the loss and current progress
        if batch % 5 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)

            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # Append the batch loss to enable calculation of the average epoch loss
        epoch_loss.append(loss)
        epoch_dice.append(dice_coefficient_multi_batch(pred, y).item())
        epoch_dice_all[batch] = dice_coefficient_multi_batch_all(pred, y).detach().tolist()
        

        # Calculate Hausdorff distance:
        # Turn model outputs from logits to onehot encoding required by hausdorff function
        pred_softmax = torch.softmax(pred, dim=1)
        print(f"Softmax shape: {pred_softmax.shape}")

        pred_onehot = torch.zeros_like(pred_softmax)
        print(f"Onehot shape: {pred_onehot.shape}")

        # Scatter ones along the class dimension in to position of the max softmax value
        pred_onehot.scatter_(1, pred_softmax.argmax(dim=1, keepdim=True), 1)
        print(f"Onehot shape after scatter: {pred_onehot.shape}")


        # Calculate Hausdorff distance for each class
        # Remove dim of 1 ground truth
        # Taking mean of Hausdorff distance of each class
        # print(torch.squeeze(y, dim=1).shape))
        
        hausdorff_distance_all = compute_hausdorff_distance(pred_onehot, 
                                                        torch.squeeze(y, dim=1),
                                                        include_background=True).detach()

        print(f"Hausdorff distance: {hausdorff_distance_all}")

        epoch_haus_loss_all[batch] = hausdorff_distance_all.tolist()[0]
        epoch_haus.append(hausdorff_distance_all.mean(dim=0))

        torch.cuda.empty_cache()

    # Calculate the average loss and accuracy for the epoch
    avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
    avg_epoch_dice =  sum(epoch_dice) / len(epoch_loss) 
    avg_epoch_haus = sum(epoch_haus) / len(epoch_loss)
    
    avg_epoch_dice_all = epoch_dice_all.mean(axis=0)
    avg_epoch_haus_loss_all = epoch_haus_loss_all.mean(axis=0)
    

    return (avg_epoch_loss, avg_epoch_dice, avg_epoch_haus, 
            avg_epoch_dice_all, avg_epoch_haus_loss_all)



# Define a validation loop function for reuse later 
def validation_loop(dataloader, device, model, loss_fn, num_classes, patch_size=None):

    print("Running validation loop...")

    # Release all unoccupied cached memory
    gc.collect()
    torch.cuda.empty_cache()
    
    valid_epoch_loss = []
    valid_epoch_dice = []
    valid_epoch_haus = []
    
    # Initialise separate dic earray which will capture dice of all tissues indvidually
    valid_epoch_dice_all = np.empty(shape=(len(dataloader), num_classes))
    valid_epoch_haus_loss_all = np.empty(shape=(len(dataloader), num_classes))

    # Set the model to evaluation mode
    model.eval()

    # Save size of dataset
    size = len(dataloader.dataset)
    print(f"Size = {size}")

    # Save number of batches
    num_batches = len(dataloader)

    # Initialise loss
    validation_loss = 0
    validation_dice = 0
    validation_hausdorff_distance = 0
    

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():

        # Loop through Xs and ys in dataloader batch
        for batch, (X, y) in enumerate(dataloader):
            
            # Load X and y to releavnt device
            X = X.to(device)
            y = y.to(device)

            if patch_size is not None:
                print(f"Performing sliding window inference with patch size: {patch_size}")
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = sliding_window_inference(X, 
                                                    roi_size=patch_size, 
                                                    sw_batch_size=2, 
                                                    predictor=model, 
                                                    overlap=0.25)
                    
            else:
                # Make predictions on the input features
                pred = model(X)

            
                

            # Determine the loss associated with the current predictions and add to batch loss
            validation_loss += loss_fn(pred, y).item()

            # Determine dice score associated with the current predictions and add to batch dice score
            validation_dice += dice_coefficient_multi_batch(pred, y).item()
            valid_epoch_dice_all[batch] = dice_coefficient_multi_batch_all(pred, y).detach().tolist()
            
            # Determine hasudorff distance associated with the current predictions and add to batch hausdorff distance
            # Turn model outputs from logits to onehot encoding required by hausdorff function
            pred_softmax = torch.softmax(pred, dim=1)
            pred_onehot = torch.zeros_like(pred_softmax)
            # Scatter ones along the class dimension in to position of the max softamx value
            pred_onehot.scatter_(1, pred_softmax.argmax(dim=1, keepdim=True), 1)
            
            # validation_hausdorff = validation_hausdorff.mean(dim=0)
            
            # Calculate Hausdorff distance
            # Remove dim of 1 ground truth
            # Validation dataloader batch size set at 2 so compute_hausdorff_distance returns tensor of shape (2, num_classes)
            valid_hausdorff_distance_all = compute_hausdorff_distance(pred_onehot, 
                                                                  torch.squeeze(y, dim=1),
                                                                  include_background=True).detach()
            
            # Add hausdorff distance for each class to batch hausdorff distance list
            valid_epoch_haus_loss_all[batch] = valid_hausdorff_distance_all.tolist()[0]
            
            # Add class mean hausdorff distance to batch hausdorff distance list
            valid_epoch_haus.append(valid_hausdorff_distance_all.mean(dim=0))
            
            # Take mean of of whole tensor (classes and batch as compute_hausdorff_distance returns tensor of shape (2, num_classes) 
            # Validation dataloader batch size set at 2, so tensor has shape (2, num_classes)
            print(f"valid_hausdorff_distance_all: {valid_hausdorff_distance_all}")
            print(f"valid_hausdorff_distance_all shape: {valid_hausdorff_distance_all.shape}")
            validation_hausdorff_distance += valid_hausdorff_distance_all.mean().item()

            torch.cuda.empty_cache()

    

    validation_loss /= num_batches
    validation_dice /= num_batches
    validation_hausdorff_distance /= num_batches

    valid_avg_epoch_dice_all = valid_epoch_dice_all.mean(axis=0)
    valid_avg_epoch_haus_loss_all = valid_epoch_haus_loss_all.mean(axis=0)

    # lr_scheduler.step(validation_loss/len(dataloader))

    print(f"""\n
        Validation Error: \n 
        Validation dice: {(100*validation_dice):>0.1f}%
        Validation dice by tissue: {valid_avg_epoch_dice_all}%
        Validation hausdorff: {validation_hausdorff_distance:>8f} \n
        Validation hausdorff by tissue: {valid_avg_epoch_haus_loss_all} \n
        Validation avg loss: {validation_loss:>8f} \n
    """)

    # Append the batch loss to enable calculation of the average epoch loss
    valid_epoch_loss.append(validation_loss)
    valid_epoch_dice.append(validation_dice)
    valid_epoch_haus.append(validation_hausdorff_distance)

    # Calculate the average loss for the epoch
    print(f"Length of valid_epoch_loss: {len(valid_epoch_loss)}")
    avg_valid_epoch_loss = sum(valid_epoch_loss) / len(valid_epoch_loss)

    # Calculate the average dice score for the epoch
    print(f"Length of valid_epoch_dice: {len(valid_epoch_dice)}")
    avg_valid_epoch_dice = sum(valid_epoch_dice) / len(valid_epoch_dice)

    # Calculate the average hausdorff distance for the epoch
    print(f"Length of valid_epoch_haus: {len(valid_epoch_haus)}")
    avg_valid_epoch_haus = sum(valid_epoch_haus) / len(valid_epoch_haus)

    return (avg_valid_epoch_loss, avg_valid_epoch_dice, avg_valid_epoch_haus, 
            valid_avg_epoch_dice_all, valid_avg_epoch_haus_loss_all) # , lr_scheduler




