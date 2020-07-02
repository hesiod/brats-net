import os
import re
import nibabel as nib
import torch
import torch.utils.data as td
import torchvision
import numpy as np

__all__ = ['BRATS']


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

        self.slice_offset_start = 40
        self.slice_offset_end = 40
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
