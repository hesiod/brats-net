import torch
import torch.utils.data as td
import torchvision
import math
import h5py

from tqdm import tqdm, trange

__all__ = ['BRATS', 'DataSplitter']


def load_sets(hdf_filename):
    hf = h5py.File(hdf_filename, "r")
    sets = []
    for v in tqdm(hf.get('imagesTr').values()):
        h = HBRATS(v)
        sets.append(h)
    return td.ConcatDataset(sets)

class BRATS(td.ConcatDataset):
    def __init__(self, hdf_filename):
        hf = h5py.File(hdf_filename, "r")
        sets = []
        for v in tqdm(hf.get('imagesTr').values()):
            h = HBRATS(v)
            sets.append(h)

        super().__init__(sets)

class HBRATS(td.Dataset):
    def __init__(self, scan_grp):
        super(HBRATS).__init__()

        self.slices = scan_grp

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        for slice_idx, slice_val in enumerate(self.slices.values()):
            if slice_idx == idx:
                slice_grp = slice_val
        assert(slice_grp is not None)

        image = slice_grp.get('image')[()]
        label = slice_grp.get('label')[()]
        image_slice = torch.from_numpy(image)
        label_slice = torch.from_numpy(label)

        return image_slice, label_slice

class DataSplitter:
    def __init__(self, hdf_filename):
        self.brats_train = BRATS(hdf_filename)

    def split_data(self, num_epochs):
        datasize = len(self.brats_train)
        data_per_epoch = int(math.floor(datasize/num_epochs))
        print('total count = {}, num_epochs = {}, per epoch = {}'.format(datasize, num_epochs, data_per_epoch))
        subsets = torch.full(size=[num_epochs], fill_value=data_per_epoch, dtype=torch.int)
        remainder = datasize - num_epochs * data_per_epoch
        if remainder > 0:
            subsets = torch.cat([subsets, torch.Tensor([remainder]).int()])
        return td.random_split(self.brats_train, subsets)


if __name__ == '__main__':
    ss = BRATS("brats_training.hdf5")
