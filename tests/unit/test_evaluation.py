# Unit tests should cover realistic use cases for your function, such as:
#     boundary cases, like the highest and lowest expected input values
#     positive, negative, zero and missing value inputs
#     examples that trigger errors that have been defined in your code

import sys
import os

if '.\\src' not in sys.path:
    sys.path.append('.\\src')

import torch
from models.evaluation import dice_coefficient


def test_dice_coefficient_same() -> None:
    """Test for dice coefficient function for perfectly overlapping masks
    """

    # Arrange
    test_pred_mask = torch.Tensor([1,1])
    test_gt_mask = torch.Tensor([1,1])
    expected_output = 1.0
    
    # Act
    output = dice_coefficient(pred_mask=test_pred_mask, gt_mask=test_gt_mask)

    # Assert
    assert output == expected_output


def test_dice_coefficient_distinct() -> None:
    """Test for dice coefficient function for completely distinct masks
    """

    # Arrange
    test_pred_mask = torch.Tensor([0,1])
    test_gt_mask = torch.Tensor([1,0])
    expected_output = 0
    
    # Act
    output = dice_coefficient(pred_mask=test_pred_mask, gt_mask=test_gt_mask)

    # Assert
    assert output == expected_output


def test_dice_coefficient_empty() -> None:
    assert 1 == 0
    


