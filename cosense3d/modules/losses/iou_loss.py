from .base_loss import BaseLoss
from cosense3d.utils.iou2d_calculator import bbox_overlaps


class IoULoss(BaseLoss):
    """
    Args:
        eps (float): Eps to avoid log(0).
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """
    def __init__(self, mode='log', eps=1e-6,  **kwargs):
        super(IoULoss, self).__init__(**kwargs)
        assert mode in ['linear', 'square', 'log']
        self.mode = mode
        self.eps = eps

    def loss(self, pred, target):
        ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=self.eps)
        if self.mode == 'linear':
            loss = 1 - ious
        elif self.mode == 'square':
            loss = 1 - ious ** 2
        elif self.mode == 'log':
            loss = -ious.log()
        else:
            raise NotImplementedError
        return loss


class GIoULoss(BaseLoss):
    """
    Args:
        eps (float): Eps to avoid log(0).
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """
    def __init__(self, eps=1e-7,  **kwargs):
        super(GIoULoss, self).__init__(**kwargs)
        self.eps = eps

    def loss(self, pred, target):
        gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=self.eps)
        loss = 1 - gious
        return loss

