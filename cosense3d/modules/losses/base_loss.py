import torch
from torch import nn


class BaseLoss(nn.Module):
    def __init__(self, reduction='mean', activation='none', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activation = activation

    @property
    def name(self):
        return self.__class__.__name__

    def loss(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, preds, targets,
                weight=None, avg_factor=None,
                reduction_override=None,
                *args, **kwargs):
        """

        Parameters
        ----------
        weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
        avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        """
        loss = self.loss(preds, targets, *args, **kwargs)
        # if weight is specified, apply element-wise weight
        if weight is not None:
            loss = loss * weight
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # if avg_factor is not specified, just reduce the loss
        if avg_factor is None:
            if reduction == 'mean':
                loss = loss.mean()
            elif reduction == 'sum':
                loss = loss.sum()
        else:
            # if reduction is mean, then average the loss by avg_factor
            if reduction == 'mean':
                # Avoid causing ZeroDivisionError when avg_factor is 0.0,
                # i.e., all labels of an image belong to ignore index.
                eps = torch.finfo(torch.float32).eps
                loss = loss.sum() / (avg_factor + eps)
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif reduction != 'none':
                raise ValueError('avg_factor can not be used with reduction="sum"')
        return self.loss_weight * loss