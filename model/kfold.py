import numpy as np
import math

__all__ = ['KFold']

# https://github.com/alejandrodebus/Pytorch-Utils

class KFold:
    def __init__(self):
        super(KFold, self).__init__()

    def get_indices(self, n_splits, length):
        '''
        Indices of the set test
        Args:
            n_splits: folds number
            length: length of the dataset
        '''
        length = int(length)
        n_splits = int(n_splits)
        fold_sizes = np.full((int(length//n_splits)), n_splits)
        indices = np.arange(length).astype(int)
        current = 0
        fold_sizes[len(fold_sizes)-1] += length%n_splits 

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
        for test_idx in self.get_indices(n_splits, length):
            train_idx = []
            for i in range(0, length):
                if not i in test_idx:
                    train_idx.append(i)
  
            yield train_idx, test_idx

   

   
    
   