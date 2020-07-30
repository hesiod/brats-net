import torch
import torch.nn as nn

import model.metrics

__all__ = ['DiceLoss', 'Loss']


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.dice_coeff = model.metrics.DiceCoefficient()

    # https://github.com/pytorch/pytorch/issues/1249
    def forward(self, pred, target):
        dc = self.dice_coeff(pred, target)
        return 1. - dc


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
