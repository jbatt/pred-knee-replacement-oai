import numpy as np
from torch.nn import functional as F
from torchvision.transforms.functional import resize

# Define debug function for printing debug statements 
def debug_print(debug=False, debug_statement="") -> None:
     if debug:
          print(debug_statement)

# Function that reads in txt file with each line in format x=y
# and converts to hyperparam dictionary
def read_hyperparams(path) -> dict:
    hyperparams = {}
    with open(path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            # Convert to float if possible, else leave as string
            try:
                value = float(value)
            except ValueError:
                pass
            hyperparams[key] = value

    return hyperparams


# Function to take MRI image, clip the pixel values to specified upper
# bound, and normalise between zero and this bound.
def clip_and_norm(image, upper_bound):
    # Clip intensity values
    image = np.clip(image, 0, upper_bound)

    # Normalize the image to the range [0, 1]
    norm = (image - 0) / (upper_bound - 0)

    return norm


# This function will crop the MRI images to a pre-chosen size.
# The lower dimension defines the top and left of the image.
# The upper dimension defines the bottom and right of the image.
# dim1_lower = top
# dim1_upper = bottom
# dim2_lower = left
# dim2_upper = right
def crop_im(image, dim1_lower, dim1_upper, dim2_lower, dim2_upper):
    # dim1_lower, dim1_upper = 120, 320
    # dim2_lower, dim2_upper = 70, 326

    cropped = image[dim1_lower:dim1_upper, dim2_lower:dim2_upper, :]

    return cropped


# This function will crop the mask to a pre-chosen size.
def crop_mask(image, dim1_lower, dim1_upper, dim2_lower, dim2_upper):
    # dim1_lower, dim1_upper = 120, 320
    # dim2_lower, dim2_upper = 70, 326

    cropped = image[:, dim1_lower:dim1_upper, dim2_lower:dim2_upper, :]

    return cropped



# This function will pad an image upto a square of a give size
def pad_to_square(x, size):
        h, w = x.shape[-2:]
        padh = size - h
        padw = size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.activated_count = 0

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif (validation_loss > (self.min_validation_loss + self.min_delta)) & self.activated_count == 0:
            self.counter += 1
            if self.counter >= self.patience:
                self.activated_count += 1
                return True
        return False
