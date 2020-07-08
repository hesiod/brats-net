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

def create_hdf(hdf_filename, data_dir, dataset_info):
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

                nifti_max = nifti_data.max(axis=(1, 2))
                for j in range(4):
                    if nifti_max[j] != 0.0:
                        nifti_data[j, :, :] = nifti_data[j, :, :] / nifti_max[j]

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
    create_hdf("brats_training.hdf5", data_dir, dataset_info)
