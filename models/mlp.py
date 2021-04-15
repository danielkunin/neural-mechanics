import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


def logistic(
    input_shape, num_classes, pretrained=False, model_dir="models",
):
    size = np.prod(input_shape)

    modules = [nn.Flatten()]
    modules.append(nn.Linear(size, num_classes))
    model = nn.Sequential(*modules)

    # Pretrained model
    if pretrained:
        pretrained_path = f"{model_dir}/logistic_mnist{num_classes}.pt"
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = pretrained_dict["model_state_dict"] # necessary because of our ckpt format
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def fc(
    input_shape, num_classes, pretrained=False, L=6, N=100, nonlinearity=nn.ReLU(), model_dir="models"
):
    size = np.prod(input_shape)

    # Linear feature extractor
    modules = [nn.Flatten()]
    modules.append(nn.Linear(size, N))
    modules.append(nonlinearity)
    for i in range(L - 2):
        modules.append(nn.Linear(N, N))
        modules.append(nonlinearity)

    # Linear classifier
    modules.append(nn.Linear(N, num_classes))
    model = nn.Sequential(*modules)

    # Pretrained model
    if pretrained:
        pretrained_path = f"{model_dir}/fc_mnist{num_classes}.pt"
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = pretrained_dict["model_state_dict"] # necessary because of our ckpt format
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def fc_bn(
    input_shape, num_classes, pretrained=False, L=6, N=100, nonlinearity=nn.ReLU(), model_dir="models"
):
    size = np.prod(input_shape)

    # Linear feature extractor
    modules = [nn.Flatten()]
    modules.append(nn.Linear(size, N))
    modules.append(nn.BatchNorm1d(N))
    modules.append(nonlinearity)
    for i in range(L - 2):
        modules.append(nn.Linear(N, N))
        modules.append(nn.BatchNorm1d(N))
        modules.append(nonlinearity)

    # Linear classifier
    modules.append(nn.Linear(N, num_classes))
    model = nn.Sequential(*modules)

    # Pretrained model
    if pretrained:
        pretrained_path = f"{model_dir}/fc-bn_mnist{num_classes}.pt"
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = pretrained_dict["model_state_dict"] # necessary because of our ckpt format
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def conv(
    input_shape,
    num_classes,
    pretrained=False,
    L=3,
    N=32,
    nonlinearity=nn.ReLU(),
    norm_layer=None,
    model_dir="models",
):
    channels, width, height = input_shape

    # Convolutional feature extractor
    modules = []
    modules.append(nn.Conv2d(channels, N, kernel_size=3, padding=3 // 2))
    modules.append(nonlinearity)
    if norm_layer is not None:
        modules.append(norm_layer(N))
    for i in range(L - 2):
        modules.append(nn.Conv2d(N, N, kernel_size=3, padding=3 // 2))
        modules.append(nonlinearity)
        if norm_layer is not None:
            modules.append(norm_layer(N))

    # Linear classifier
    modules.append(nn.Flatten())
    modules.append(nn.Linear(N * width * height, num_classes))
    model = nn.Sequential(*modules)

    # Pretrained model
    if pretrained:
        pretrained_path = f"{model_dir}/conv_mnist{num_classes}.pt"
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = pretrained_dict["model_state_dict"] # necessary because of our ckpt format
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model
