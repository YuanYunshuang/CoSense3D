import torch
from .base_loss import BaseLoss


class L1Loss(BaseLoss):
    """
    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def loss(self, pred, target):
        if target.numel() == 0:
            return pred.sum() * 0

        assert pred.size() == target.size()
        loss = torch.abs(pred - target)
        return loss


class SmoothL1Loss(BaseLoss):
    """
    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """
    def __init__(self, beta=1.0, **kwargs):
        super(SmoothL1Loss, self).__init__(**kwargs)
        assert beta > 0
        self.beta = beta

    def loss(self, pred, target):
        if target.numel() == 0:
            return pred.sum() * 0

        assert pred.size() == target.size()
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta,
                           0.5 * diff * diff / self.beta,
                           diff - 0.5 * self.beta)
        return loss