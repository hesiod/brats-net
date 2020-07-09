import torch
import torch.utils.data as td
import torchvision
import math
import h5py

from tqdm import tqdm, trange

__all__ = ['BRATS', 'DataSplitter']


class BRATS(td.ConcatDataset):
    def __init__(self, hdf_filename):
        self.hdf_filename = hdf_filename

        with h5py.File(self.hdf_filename, 'r') as hf:
            sets = []
            for v in tqdm(hf.get('imagesTr').values()):
                h = HBRATS(self, v)
                sets.append(h)

        self.hdf_file = None

        super().__init__(sets)

class HBRATS(td.Dataset):
    def __init__(self, parent, scan_grp):
        super(HBRATS).__init__()


        self.parent = parent

        self.scan_name = scan_grp.name
        self.slice_count = len(scan_grp)
        self.slices = None

    def __len__(self):
        return self.slice_count

    def __getitem__(self, idx):
        if self.parent.hdf_file is None:
            self.parent.hdf_file = h5py.File(self.parent.hdf_filename, 'r')
        if self.slices is None:
            self.slices = self.parent.hdf_file.get(self.scan_name)

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
