import torch

__all__ = ['Loss']


# https://github.com/pytorch/pytorch/issues/1249
def dice_loss(pred, target):
    smooth = 1.

    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([5.0]))
    
    def forward(self, predicted, target):
        bce_loss = self.bce(predicted, target)
        dice_loss = dice_loss(predicted, target)

        return bce_loss + dice_loss
