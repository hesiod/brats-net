import numpy as np

__all__ = ['KFold']

# https://github.com/alejandrodebus/Pytorch-Utils

class KFold:
    def __init__(self):
        super(KFold, self).__init__()
        pass

    def partitions(self, length, k):
        '''
        Distribution of the folds
        Args:
            length: length of the dataset
            k: folds number
        '''
        n_partitions = np.ones(k) * int(length/k)
        n_partitions[0:(length % k)] += 1
        return n_partitions

    def get_indices(self, n_splits, length):
        '''
        Indices of the set test
        Args:
            n_splits: folds number
            length: length of the dataset
        '''
        l = self.partitions(self, length, n_splits)
        fold_sizes = l * length
        indices = np.arange(length).astype(int)
        current = 0
        for fold_size in fold_sizes:
            start = current
            stop =  current + fold_size
            current = stop
            yield(indices[int(start):int(stop)])

    def k_folds(self, n_splits, length):
        '''
        Generates folds for cross validation
        Args:
            n_splits: folds number
            length: length of the dataset
        '''
        indices = np.arange(length).astype(int)
        for test_idx in self.get_indices(self, n_splits, length):
            train_idx = np.setdiff1d(indices, test_idx)
            yield train_idx, test_idx

   

   
    
   