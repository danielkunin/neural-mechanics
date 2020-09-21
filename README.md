# Neural Mechanics

# To Do:
- Condense models and remove model class?
- Add flag to save +/- k from each checkpoint
- Handle the PT_MODELS map from pytorch names to labels better?
- Get the argparser from utils/flags.py in the visualization scripts


## Getting Started
First clone this repo, then install all dependencies
```
pip install -r requirements.txt
```
The code was tested with Python 3.6.0.

## Tracking Dynamics
Below is a description of the major sections of the code base. Run `python main.py --help` for a complete description of flags and hyperparameters.

### Training

#### Datasets
This code base supports the following datasets: MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet, ImageNet. 

All datasets except Tiny ImagNet and ImageNet will download automatically.  For Tiny ImageNet, download the data directly from [https://tiny-imagenet.herokuapp.com](https://tiny-imagenet.herokuapp.com), move the unzipped folder ``tiny-imagnet-200`` into the ```Data``` folder, run the script `python Utils/tiny-imagenet-setup.py` from the home folder. For ImageNet setup locally in the ```Data``` folder.

#### Models

There are three model classes each defining a variety of model architectures:
 - Default models support basic dense and convolutional model.
 - Tiny ImageNet models support VGG/ResNet architectures based on this Github [repository](https://github.com/weiaicunzai/pytorch-cifar100).
 - ImageNet models supports VGG/ResNet architectures from [torchvision](https://pytorch.org/docs/stable/torchvision/models.html).


### Extraction

### Visualization


## Citation
If you use this code for your research, please cite...
