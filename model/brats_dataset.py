import os
import re
import nibabel as nib
import torch
import torch.utils.data as td
import torchvision
import numpy as np
import math

__all__ = ['BRATS', 'DataSplitter']


class BRATS(td.Dataset):
    def __init__(self, data_dir, train):
        super(BRATS).__init__()
        self.train = train
        self.data_dir = data_dir
        self.data = []
        self.filenames = []
        match_brats_filename = re.compile(r"^BRATS_[0-9]{3}.nii.gz$")
        if self.train:
            self.labels = []
            image_dir = os.path.join(self.data_dir, 'imagesTr')
            label_dir = os.path.join(self.data_dir, 'labelsTr')
        else:
            image_dir = os.path.join(self.data_dir, 'imagesTs')

        for fn in os.listdir(image_dir):
            abs_fn = os.path.join(image_dir, fn)
            if os.path.isfile(abs_fn):
                # print(fn)
                if match_brats_filename.match(fn):
                    nifti_img = nib.load(abs_fn)
                    # nifti_data = nifti_img.get_fdata()
                    self.data.append(nifti_img)
                    self.filenames.append(abs_fn)

        self.slice_offset_start = 10
        self.slice_offset_end = 10
        self.slice_count = 155 - self.slice_offset_start - self.slice_offset_end

        self.transform_cache = {}

        self.num_files = len(self.data)
        print('Loaded {} files'.format(self.num_files))
        if self.train:
            for fn in os.listdir(image_dir):
                abs_fn = os.path.join(label_dir, fn)
                if os.path.isfile(abs_fn) and match_brats_filename.match(fn):
                    nifti_img = nib.load(abs_fn)
                    self.labels.append(nifti_img)
            print('Loaded {} labels'.format(len(self.labels)))

    def __len__(self):
        return self.num_files * self.slice_count

    def __getitem__(self, idx):
        sample_index = idx // self.slice_count
        slice_index = self.slice_offset_start + idx % self.slice_count

        nifti_data = np.asarray(self.data[sample_index].dataobj[:, :, slice_index, :])
        nifti_slice = torch.from_numpy(np.copy(nifti_data)).transpose(0, 2)

        if idx in self.transform_cache:
            nifti_normalize = self.transform_cache[idx]
        else:
            nifti_norm = np.linalg.norm(nifti_slice)
            if (nifti_norm > 0.0):
                nifti_mean = torch.mean(nifti_slice, dim=(1, 2)) + 1e-9
                nifti_std = torch.std(nifti_slice, dim=(1, 2)) + 1e-9
                nifti_normalize = torchvision.transforms.Normalize(nifti_mean, nifti_std)
            else:
                nifti_normalize = None
            self.transform_cache[idx] = nifti_normalize

        if nifti_normalize is not None:
            nifti_slice = nifti_normalize(nifti_slice)

        if self.train:
            label_data = np.asarray(self.labels[sample_index].dataobj[:, :, slice_index])
            label_slice = torch.from_numpy(np.copy(label_data)).transpose(0, 1)

        return nifti_slice, label_slice


class DataSplitter:
    def __init__(self, data_folder='Task01_BrainTumour'):
        self.brats_train = BRATS(data_folder, train=True)
        self.brats_test = BRATS(data_folder, train=False)
        # self.train_iter = td.DataLoader(brats_train, batch_size, shuffle=True, num_workers=num_workers)
        # self.test_iter = td.DataLoader(brats_test, batch_size, shuffle=False, num_workers=num_workers)

    def split_data(self, num_epochs):
        datasize = len(self.brats_train)
        data_per_epoch = int(math.floor(datasize/num_epochs))
        print('total count = {}, num_epochs = {}, per epoch = {}'.format(datasize, num_epochs, data_per_epoch))
        subsets = torch.full(size=[num_epochs], fill_value=data_per_epoch, dtype=torch.int)
        remainder = datasize - num_epochs * data_per_epoch
        if remainder > 0:
            subsets = torch.cat([subsets, torch.Tensor([remainder]).int()])
        return td.random_split(self.brats_train, subsets)
