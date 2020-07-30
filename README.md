# Medical Image Segmentation: Brain Tumours
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hesiod/brats-net/blob/master/unet.ipynb)

## Project structure

- **`unet.ipynb`**: Demo Jupyter notebook
- `train.py`: Main training program
- `prepare_data.py`: Data preprocessing and HDF5 generation script (see below)
- `model`
  - `brats_dataset.py`: `BRATS` dataset class that allows loading the HDF5 dataset
  - `loss.py`: Loss functions
  - `metrics.py`: Dice coefficient and Jaccard index as metrics
  - `res_unet.py`: Residual variant U-Net
  - `unet.py`: Classical, non-residual U-Net
  - `utils.py`: Utility functions

## Preparing the dataset
The training program `train.py` requires the original dataset from the
[Medical Decathlon](http://medicaldecathlon.com/)
to be transformed and prepared using the preparation script `prepare_data.py`.
The script expects a folder name with the following content:

1. A dataset metadata file called `dataset.json`
1. Folders `imagesTr` and `labelsTr` containing training images and labels

Datasets conforming to that structure can be downloaded from the
Medical Decathlon website.

The script performs normalization of the channel data, culls empty slices and
slices without labeled features and writes the remaining transformed
slices into an HDF file.

**Note**: Running the data preparation script
can take up to multiple hours
depending on system I/O performance.

## Training configuration
Adapt the provided sample configuration file `params.sample.json`
to suit your needs.

Parameter description:
 - `meta_name`: Name of this training run
 - `model`: `UNet` for the non-residual or `UResNet` for the residual variant
 - `input_channels`: Number of input channels (4 for BraTS)
 - `gradient_clip_value`: Gradient clip value
 - `batch_size`: Batch size
 - `epoch_size`: Number of slices per epoch
 - `num_workers`: Number of data loader workers
 - `num_epochs`: Number of epochs
 - `lr`: Learning Rate
 - `optimizer`: `AdamW` or `SGD`
 - `sgd_momentum`: Momentum for `SGD`
 - `sgd_weight_decay`: Weight decay for `SGD`
 - `adam_weight_decay`: Weight decay for `AdamW`
 - `scheduler`: `CosineAnnealingWarmRestarts` or `ReduceLROnPlateau`
 - `sgdr_initial_period`: Initial period for the `CosineAnnealingWarmRestarts` scheduler (`T_0` in the paper)
 - `sgdr_period_multiplier`: Period length multiplier for the `CosineAnnealingWarmRestarts` scheduler (`T_mult` in the paper)
 - `lr_scheduler_patience`: Patience for the `ReduceLROnPlateau` scheduler (number of epochs without test loss improvement to wait before reducing learning rate)

## Training the model
Run `train.py` with the following parameters:

- `--params`: Training configuration (see above)
- `--checkpoint` (optional): Load model from training checkpoint
- As the last parameter, include the HDF5 dataset

Example invocation:
```
python3 ./train.py --params params.sample.json dataset_BRATS.hdf5
```

## Relevant papers

### 2D U-Net-like
* U-Net: Convolutional Networks for Biomedical Image Segmentation, https://arxiv.org/abs/1505.04597, https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
* nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation, https://arxiv.org/abs/1809.10486
* W-Net: A Deep Model for Fully Unsupervised Image Segmentation, https://arxiv.org/abs/1711.08506

### 3D U-Net-like
* 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation, https://arxiv.org/abs/1606.06650
* V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, https://arxiv.org/abs/1606.04797
