import torch

__all__ = ['try_gpu']

# From d2l.ai
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

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
        print(acc_mean)
    # Return mean of batch
    return acc_mean