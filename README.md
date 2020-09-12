# Neural Mechanics


## Getting Started
First clone this repo, then install all dependencies
```
pip install -r requirements.txt
```
The code was tested with Python 3.6.0.

## Code Base
Below is a description of the major sections of the code base. Run `python main.py --help` for a complete description of flags and hyperparameters.

### Datasets
This code base supports the following datasets: MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet, ImageNet. 

All datasets except Tiny ImagNet and ImageNet will download automatically.  For Tiny ImageNet, download the data directly from [https://tiny-imagenet.herokuapp.com](https://tiny-imagenet.herokuapp.com), move the unzipped folder ``tiny-imagnet-200`` into the ```Data``` folder, run the script `python Utils/tiny-imagenet-setup.py` from the home folder. For ImageNet setup locally in the ```Data``` folder.

### Models

There are three model classes each defining a variety of model architectures:
 - Default models support basic dense and convolutional model.
 - Tiny ImageNet models support VGG/ResNet architectures based on this Github [repository](https://github.com/weiaicunzai/pytorch-cifar100).
 - ImageNet models supports VGG/ResNet architectures from [torchvision](https://pytorch.org/docs/stable/torchvision/models.html).


### Experiments

Below is a list and description of the experiment files found in the `Experiment` folder:
 - `example.py`: used to test code.
 - `singleshot.py`: used to make figure 1, 2, and 6.
 - `multishot.py`: used to make figure 5a.
 - `unit-conservation.py`: used to make figure 3.
 - `layer-conservation.py`: used to make figure 4.
 - `lottery-layer-conservation.py`: used to make figure 5b.
 - `synaptic-flow-ratio.py`: used to make figure 7.


### Results

All data used to generate the figures in our paper can be found in the `Results/data` folder.

## Citation
If you use this code for your research, please cite...
