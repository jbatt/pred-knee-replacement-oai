# Define functions to help with model training
import gc
import torch
import numpy as np

# from src.metrics.evaluation import dice_coefficient_multi_batch, dice_coefficient_multi_batch_all
from metrics.metrics import dice_coefficient_multi_batch, dice_coefficient_multi_batch_all

from monai.losses.hausdorff_loss import HausdorffDTLoss


# Define a training loop function for reuse later 
def train_loop(
        dataloader, 
        device, 
        model, 
        loss_fn, 
        optimizer, 
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
        print("Computing model predictions...")
        pred = model(X)
        print(f"Prediction shape: {pred.shape}")

        print("Calculating loss...")
        loss = loss_fn(pred, y)

        # Backpropagation
        print("Computing backpropagation...")
        loss.backward()
        
        # Perform model step
        optimizer.step()
        # Reset gradients to zero
        optimizer.zero_grad()
        
        # Store batch size
        batch_size = len(y)

        # # For every 5th batch, print the loss and current progress
        if batch % 5 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)

            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # # Determine number of correct predictions for top 1 accuracy
        # correct += (pred.argmax(1) == y).type(torch.float).sum().item()


        # Append the batch loss to enable calculation of the average epoch loss
        epoch_loss.append(loss)
        epoch_dice.append(dice_coefficient_multi_batch(pred, y).item())
        epoch_dice_all[batch] = dice_coefficient_multi_batch_all(pred, y).detach().tolist()
        epoch_haus_loss_all[batch] = HausdorffDTLoss(pred, y, softmax=True).detach().tolist()

    # Calculate the average loss and accuracy for the epoch
    avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
    avg_epoch_dice =  sum(epoch_dice) / len(epoch_loss) 
    avg_epoch_dice_all = epoch_dice_all.mean(axis=0)
    avg_epoch_haus_loss_all = epoch_haus_loss_all.mean(axis=0)
    

    return avg_epoch_loss, avg_epoch_dice, avg_epoch_dice_all, avg_epoch_haus_loss_all



# Define a validation loop function for reuse later 
def validation_loop(dataloader, device, model, loss_fn, num_classes):

    print("Running validation loop...")

    # Release all unoccupied cached memory
    gc.collect()
    torch.cuda.empty_cache()
    
    valid_epoch_loss = []
    valid_epoch_dice = []
    
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
    

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():

        # Loop through Xs and ys in dataloader batch
        for batch, (X, y) in enumerate(dataloader):
            
            # Load X and y to releavnt device
            X = X.to(device)
            y = y.to(device)

            # Make predictions on the input features
            pred = model(X)

            # Determine the loss associated with the current predictions and add to batch loss
            validation_loss += loss_fn(pred, y, num_classes).item()

            # Determine dice score associated with the current predictions and add to batch dice score
            validation_dice += dice_coefficient_multi_batch(pred, y).item()
            valid_epoch_dice_all[batch] = dice_coefficient_multi_batch_all(pred, y).detach().tolist()
            valid_epoch_haus_loss_all[batch] = HausdorffDTLoss(pred, y, softmax=True).detach().tolist()

    validation_loss /= num_batches
    validation_dice /= num_batches
    valid_avg_epoch_dice_all = valid_epoch_dice_all.mean(axis=0)
    valid_avg_epoch_haus_loss_all = valid_epoch_haus_loss_all.mean(axis=0)


    # lr_scheduler.step(validation_loss/len(dataloader))

    print(f"""\n
          Validation Error: \n 
          Validation dice: {(100*validation_dice):>0.1f}%
          Validation dice by tissue: {valid_avg_epoch_dice_all}%
          Validation avg loss: {validation_loss:>8f} \n
    """)

    # Append the batch loss to enable calculation of the average epoch loss
    valid_epoch_loss.append(validation_loss)
    valid_epoch_dice.append(validation_dice)

    # Calculate the average loss for the epoch
    avg_valid_epoch_loss = sum(valid_epoch_loss) / len(valid_epoch_loss)

    # Calculate the average dice score for the epoch
    avg_valid_epoch_dice = sum(valid_epoch_dice) / len(valid_epoch_dice)

    return avg_valid_epoch_loss, avg_valid_epoch_dice, valid_avg_epoch_dice_all, valid_avg_epoch_haus_loss_all # , lr_scheduler




