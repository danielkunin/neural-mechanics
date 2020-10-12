# Neural Mechanics

## To Do:
- Condense models and remove model class?
- Add flag to save +/- k from each checkpoint
- Get the argparser from utils/flags.py in the visualization scripts
- Move all bash scripts into a script folder and find ways to execute from here?
- Make jupyter notebook interface with plotting make more sense and make outer folder visualize.py?
- Add plotting of loss function and training curves

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
1. Ensure you have the appropriate tooling installed. If you installed this on gcloud, you should. It doesnçt hurt to check:
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

TODO

### Caching metrics

TODO

### Visualization

TODO

## Citation
If you use this code for your research, please cite...
