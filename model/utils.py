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
    acc= []

    with torch.no_grad():
        for i in range(prediction.shape[0]):
            intersection = torch.logical_and(prediction[i::], truth[i::])
            union = torch.logical_or(prediction[i::], truth[i::])
            print(prediction[i::].size())
            print(truth[i::].size())
            print(prediction[i::])
            print(truth[i::])
            #print(torch.sum(intersection))
            #print(torch.sum(union))
            acc.append(torch.sum(intersection).double() / torch.sum(union).double())

    # Return mean of batch
    return sum(acc)/len(acc)