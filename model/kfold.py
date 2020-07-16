import numpy as np

__all__ = ['KFold']

class KFold:
    def __init__(self):
        super(KFold, self).__init__()
        pass

    def partitions(self, amount_patients, k):
        '''
        Distribution of the folds
        Args:
            amount_patients: number of patients
            k: folds number
        '''
        n_partitions = np.ones(k) * int(amount_patients/k)
        n_partitions[0:(amount_patients % k)] += 1
        return n_partitions

    def get_indices(self, n_splits = 3, length = 0):
        '''
        Indices of the set test
        Args:
            n_splits: folds number
            amount_patients: number of patients
            frames: length of the sequence of each patient
        '''
        l = self.partitions(self, amount_patients, n_splits)
        fold_sizes = l * frames
        indices = np.arange(length).astype(int)
        current = 0
        for fold_size in fold_sizes:
            start = current
            stop =  current + fold_size
            current = stop
            yield(indices[int(start):int(stop)])

    def k_folds(self, n_splits = 3, length):
        '''
        Generates folds for cross validation
        Args:
            n_splits: folds number
            amount_patients: number of patients
            frames: length of the sequence of each patient
        '''
        indices = np.arange(length).astype(int)
        for test_idx in self.get_indices(self, n_splits, length):
            train_idx = np.setdiff1d(indices, test_idx)
            yield train_idx, test_idx

   

   
    
   