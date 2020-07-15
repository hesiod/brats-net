import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Loss']


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.

    # https://github.com/pytorch/pytorch/issues/1249
    def forward(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        p = (2. * intersection + self.smooth)
        q = (iflat.sum() + tflat.sum() + self.smooth)
        return (-1)*(p/q)


class Loss(nn.Module):
    def __init__(self, pos_weight):
        super(Loss, self).__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.sigmoid = nn.Sigmoid()
        self.dice_loss = DiceLoss()

    def forward(self, predicted, target):
        bce = self.bce_loss(predicted, target)
        predicted_sm = self.sigmoid(predicted)
        dice = self.dice_loss(predicted_sm, target)

        return bce + dice
