import os
import nibabel as nib
import numpy as np
import json
import h5py

from tqdm import tqdm, trange

__all__ = ['BRATS', 'DataSplitter']

def load_dataset_metadata(data_dir):
    dataset_path = os.path.join(data_dir, 'dataset.json')
    with open(dataset_path) as dataset_file:
        dataset_info = json.loads(dataset_file.read())
    return dataset_info

def calc_means(data_dir, dataset_info):
    slices_per_file = 155
    files = dataset_info.get('training')
    file_count = len(files)
    means = np.empty(shape=(file_count, 4))
    stddev = np.empty_like(means)
    for fidx, f in enumerate(tqdm(files, desc='scan', position=0)):
        nifti_fn = os.path.join(data_dir, f.get('image'))
        nifti_img = nib.load(nifti_fn)
        nifti_alldata = nifti_img.get_fdata(caching='unchanged')

        image_masked = np.ma.masked_less(nifti_alldata, 1e-3)
        means[fidx, :] = image_masked.mean(axis=(0, 1, 2))
        stddev[fidx, :] = image_masked.std(axis=(0, 1, 2))

        nifti_img.uncache()

    means = means.mean(axis=0)
    stddev = stddev.mean(axis=0)

    print('per-channel mean: {}'.format(means))
    print('per-channel standard deviation: {}'.format(stddev))

def create_hdf(hdf_filename, data_dir, dataset_info):
    mean =   np.array([460.91295331, 607.31359224, 604.41613027, 481.6058893])
    stddev = np.array([151.81789154, 150.64656103, 157.67084461, 183.92931095])

    with h5py.File(hdf_filename, "a") as hf:
        slices_per_file = 155
        files = dataset_info.get('training')
        file_count = len(files)
        for f in tqdm(files, desc='scan', position=0):
            grp = hf.create_group(f.get('image'))
            nifti_fn = os.path.join(data_dir, f.get('image'))
            nifti_img = nib.load(nifti_fn)
            label_fn = os.path.join(data_dir, f.get('label'))
            label_img = nib.load(label_fn)
            nifti_alldata = nifti_img.get_fdata(caching='unchanged')
            label_alldata = label_img.get_fdata(caching='unchanged')

            for i in range(slices_per_file):
                nifti_data = nifti_alldata[:, :, i, :]
                nifti_data = nifti_data.transpose(2, 1, 0)
                label_data = label_alldata[:, :, i]
                label_data = label_data.transpose(1, 0)

                img_norm = np.linalg.norm(nifti_data)
                label_norm = np.linalg.norm(label_data)
                if img_norm < 1e-3 or label_norm < 1e-3:
                    continue

                for j in range(4):
                    nifti_data[j, :, :] = (nifti_data[j, :, :] - mean[j]) / stddev[j]

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
    data_dir = 'Task01_BrainTumour'
    dataset_info = load_dataset_metadata(data_dir)
    calc_means(data_dir, dataset_info)
    create_hdf("brats_training.hdf5", data_dir, dataset_info)
