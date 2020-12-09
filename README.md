# Neural Mechanics

## Getting Started
:snake: Python 3.6+

First clone this repo, then install all dependencies
```
pip install -r requirements.txt
```

### Style :computer:
1. As much as possible, format code with [`black`](https://black.readthedocs.io/en/stable/)
1. `pre_commit.sh` is provided to ensure that 1) code is formatted with `black` and 2) all unit tests pass. If you want to run this script as a pre-commit hook, i.e., commits will be blocked if formatting or tests fail, make a file with the following code snipped in `.git/hooks/pre-commit`:
    ```
    #!/bin/sh

    echo "# Running pre-commit hook"
    echo "#########################"
    ./pre_commit.sh
    if [ $? -ne 0 ]; then
        echo "At least one test failed, aborting commit"
        exit 1
    fi
    ```

## Training a model and tracking its dynamics
Training a model is as easy as running `python train.py` with the appropriate flags.
Below is a description of the major sections of the code base. Run `python train.py --help` for a complete description of flags and hyperparameters.

#### Datasets
This code base supports the following datasets: MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet, ImageNet.

All datasets except Tiny ImagNet and ImageNet will download automatically.  For Tiny ImageNet, download the data directly from [https://tiny-imagenet.herokuapp.com](https://tiny-imagenet.herokuapp.com), move the unzipped folder ``tiny-imagnet-200`` into the ```Data``` folder, run the script `python Utils/tiny-imagenet-setup.py` from the home folder. For ImageNet setup locally in the ```Data``` folder.

#### Models

There are three model classes each defining a variety of model architectures:
 - Default models support basic dense and convolutional model.
 - Tiny ImageNet models support VGG/ResNet architectures based on this Github [repository](https://github.com/weiaicunzai/pytorch-cifar100).
 - ImageNet models supports VGG/ResNet architectures from [torchvision](https://pytorch.org/docs/stable/torchvision/models.html).

#### TPU training support
Training on TPU is supported but requires additional configuration.

1. Create a compute instance in google cloud following [these instructions](https://cloud.google.com/tpu/docs/tutorials/resnet-pytorch). It is important to use the `torch-xla` image from the `ml-images` family, as this comes with `conda` environments preconfigured to work with `torch-xla`.
1. Create a TPU device. To facilitate this process, `scripts/make_tpus.sh` is provided. You can modify naming and IP address ranges to suit your needs.
1. Ensure you have the appropriate tooling installed. If you installed this on gcloud, you should. It doesn't hurt to check:
    ```bash
    pip install --upgrade google-api-python-client
    pip install --upgrade oauth2client
    pip install google-compute-engine
    ```
1. Ensure you provide a Google Cloud Storage bucket via `--save-dir=gs://my-bucket-name` to avoid overfilling your instance drive with checkpoints and metrics.
1. From a usage standpoint, you only need to specify the `--tpu` flag with the name of the TPU device you want to run on and `--workers` set to the number of cores your TPU setup has. This number is 8 for single V3-8 TPU devices (TPU pod support coming soon!).
    1. If this is the first time running on TPU, you'll need to get the datasets locally on the TPU device. For now, start a training run without the `--tpu` flag to avoid multiprocessing race conditions. You can abort it once the data has been downloaded. For ImageNet we are working on having a disk you can readily clone in gcloud, but for now it involves an pproximately 3 hour process of copying 150 GB over to the compute instance.
1. Once your training finished (or even halfway during training) you can use the `scripts/sync_gcloud.sh` script on a local machine (with the `gcloud-cli` installed) to copy the collected data over for analysis and plotting. Modify to suit your needs.

Note: while training on TPU, if your process dies unexpectedly or you force quit it, sometimes ghost processes will persist and keep the TPU device busy. `scripts/kill_all.sh` is provided to wipe such processes from the instance after such an event. Modify appropriately.

### Extraction

After the model has been trained using the `train.py` script, we run an intermediate feature extraction phase which reads in checkpoints saved during training and extracts the evaluation metrics, weights, biases and optimizer buffers for the relevant metrics.

This is precisely the `extract.py` script and needs only be pointed to the experiment, expid and directory where that experiment's directory can be found (if changed from the default during training).
A full list of flags can be obtained through the `--help` option.

### Caching metrics

Once features have been extracted from the checkpoints, the interesting weight metrics along with their theoretical predictions are computed.
A cache of the computed metrics is stored with the idea that visualzation will be left to the end user for instance in a notebook.
Such a user can simply load the cache file and only worry about displaying the data.

You guessed it, the `cache.py` script provides this functionality with similar syntax as `extract.py` in terms of flags.
It takes an optional additional flag `--metrics` which takes a comma separated list of the metrics to generate a cache for.
If the flag is not provided, caches for all metrics are saved.
It is particularly useful for recomputing a single cache or computing a cache for a newly added metric.

### Visualization

Visualization of the metrics is intended to be done by the end user.
However we provide:
- The `plot.py` script with basic plotting functionality using the above generated caches.
- The `notebooks/plots.ipynb` notebook which shows how the caches might be used to quickly iterate and fine-tune plots. This is the notebook used to generate the empirical plots in the original paper.

## Citation
If you use this code for your research, please cite...
