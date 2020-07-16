import os
import sys
import nibabel as nib
import numpy as np
import json
import h5py

from tqdm import tqdm, trange

__all__ = ['DataPrep']

def load_dataset_metadata(data_dir):
    dataset_path = os.path.join(data_dir, 'dataset.json')
    with open(dataset_path) as dataset_file:
        dataset_info = json.loads(dataset_file.read())
    return dataset_info

class DataPrep():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dataset_info = load_dataset_metadata(data_dir)
        self.channel_mean = None
        self.channel_stddev = None

    def calc_means(self):
        files = self.dataset_info.get('training')
        file_count = len(files)
        means = np.empty(shape=(file_count, 4))
        stddev = np.empty_like(means)
        for fidx, f in enumerate(tqdm(files, desc='scan', position=0)):
            nifti_fn = os.path.join(self.data_dir, f.get('image'))
            nifti_img = nib.load(nifti_fn)
            nifti_alldata = nifti_img.get_fdata(caching='unchanged')

            # If source data does not have a dedicated channel dimension, add one
            if len(nifti_alldata.shape) < 4:
                    nifti_alldata = np.expand_dims(nifti_alldata, axis=3)

            image_masked = np.ma.masked_less(nifti_alldata, 1e-3)
            means[fidx, :] = image_masked.mean(axis=(0, 1, 2))
            stddev[fidx, :] = image_masked.std(axis=(0, 1, 2))

            nifti_img.uncache()

        self.channel_mean = means.mean(axis=0)
        self.channel_stddev = stddev.mean(axis=0)

        print('per-channel mean: {}'.format(self.channel_mean))
        print('per-channel standard deviation: {}'.format(self.channel_stddev))

    def create_hdf(self, hdf_filename=None):
        if self.channel_mean is None:
            self.channel_mean =   np.array([460.91295331, 607.31359224, 604.41613027, 481.6058893])
        if self.channel_stddev is None:
            self.channel_stddev = np.array([151.81789154, 150.64656103, 157.67084461, 183.92931095])
        if hdf_filename is None:
            hdf_filename = 'dataset_{}.hdf5'.format(self.dataset_info.get('name'))

        with h5py.File(hdf_filename, "a") as hf:
            hf.attrs['name'] = self.dataset_info.get('name')
            hf.attrs['description'] = self.dataset_info.get('description')

            files = self.dataset_info.get('training')
            for f in tqdm(files, desc='scan', position=0):
                grp = hf.create_group(f.get('image'))
                nifti_fn = os.path.join(self.data_dir, f.get('image'))
                nifti_img = nib.load(nifti_fn)
                label_fn = os.path.join(self.data_dir, f.get('label'))
                label_img = nib.load(label_fn)
                nifti_alldata = nifti_img.get_fdata(caching='unchanged')
                label_alldata = label_img.get_fdata(caching='unchanged')

                # If source data does not have a dedicated channel dimension, add one
                if len(nifti_alldata.shape) < 4:
                    nifti_alldata = np.expand_dims(nifti_alldata, axis=3)

                for i in range(nifti_alldata.shape[2]):
                    nifti_data = nifti_alldata[:, :, i, :]
                    nifti_data = nifti_data.transpose(2, 1, 0)
                    label_data = label_alldata[:, :, i]
                    label_data = label_data.transpose(1, 0)

                    img_norm = np.linalg.norm(nifti_data)
                    label_norm = np.linalg.norm(label_data)
                    if img_norm < 1e-3 or label_norm < 1e-3:
                        continue

                    for j in range(nifti_alldata.shape[3]):
                        nifti_data[j, :, :] = (nifti_data[j, :, :] - self.channel_mean[j]) / self.channel_stddev[j]

                    slice_grp = grp.create_group('slice_{}'.format(i))
                    slice_grp.attrs['slice_index'] = i
                    img_ds = slice_grp.create_dataset(
                        'image',
                        data=nifti_data,
                        compression="gzip",
                        chunks=(1, 240, 240),
                        shuffle=True
                        )
                    label_ds = slice_grp.create_dataset(
                        'label',
                        data=label_data,
                        compression="gzip",
                        shuffle=True
                        )

                hf.flush()
                nifti_img.uncache()
                label_img.uncache()


if __name__ == '__main__':
    if sys.argv[1] is not None:
        data_dir = sys.argv[1]
    else:
        data_dir = 'Task01_BrainTumour'
    prep = DataPrep(data_dir)
    prep.calc_means()
    prep.create_hdf()
