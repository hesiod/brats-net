import torch
import torch.nn as nn

__all__ = ['DiceCoefficient', 'jaccard']

class DiceCoefficient(nn.Module):
    def __init__(self):
        super(DiceCoefficient, self).__init__()
        self.smooth = 1.

    # https://github.com/pytorch/pytorch/issues/1249
    def forward(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        p = 2. * intersection + self.smooth
        q = iflat.sum() + tflat.sum() + self.smooth
        return p / q

# Jaccard metric is defined by the ratio of pixels in the prediction AND the ground truth (intersection) to the pixels in at least one of these (union)
def jaccard(prediction, truth):
    assert(prediction.shape == truth.shape)

    with torch.no_grad():
        batch_count = prediction.shape[0]
        acc = torch.empty(batch_count)
        for i in range(batch_count):
            intersection = torch.logical_and(prediction[i, ...], truth[i, ...])
            union        = torch.logical_or (prediction[i, ...], truth[i, ...])
            acc[i] = torch.sum(intersection).double() / torch.sum(union).double()
        acc_mean = acc.mean().item()
    # Return mean of batch
    return acc_mean