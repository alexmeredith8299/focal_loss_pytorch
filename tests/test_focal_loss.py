import pytest
from focal_loss_pytorch import BinaryFocalLoss
import torch

def test_init_focal_loss():
    """
    Ensure we can initialize FocalLoss() object
    with no issues.
    """
    loss_fn = BinaryFocalLoss(reduction='none')
    loss_fn = BinaryFocalLoss(reduction='mean')
    loss_fn = BinaryFocalLoss(reduction='sum')
    assert True

def test_perfect_prediction():
    """
    Ensure loss is zero if prediction is correct.
    """
    loss_fn = BinaryFocalLoss(reduction='mean')
    input_tensor = torch.Tensor([1, 0])
    target = torch.Tensor([1, 0])
    loss = loss_fn(input_tensor, target)
    assert loss==0

def test_imperfect_prediction():
    """
    Ensure loss matches expected for incorrect predictions.
    """
    loss_no_reduce = BinaryFocalLoss(reduction='none')
    loss_sum = BinaryFocalLoss(reduction='sum')
    loss_mean = BinaryFocalLoss(reduction='mean')

    input_tensor = torch.Tensor([[[[0.17, 0.17], [0.92, 0.83]]]])
    target = torch.Tensor([[[[1, 0], [0, 1]]]])

    loss = loss_no_reduce(input_tensor, target)
    assert torch.max(torch.abs(loss-torch.Tensor([[[[1.22070, 0.00538492], [2.13778, 0.00538492]]]]))) < 1e-5

    loss = loss_sum(input_tensor, target)
    assert torch.abs(loss-3.369247) < 1e-5

    loss = loss_mean(input_tensor, target)
    assert torch.abs(loss-0.842312) < 1e-5

