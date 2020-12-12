import torch
import numpy as np
from torchvision import datasets, transforms
import torch.optim as optim
from models import mlp
from models import tinyimagenet_vgg
from models import tinyimagenet_resnet
from models import imagenet_vgg
from models import imagenet_resnet
from optimizers import custom_sgd
from optimizers import lamb
from utils import custom_datasets


def configure_tpu(tpu_name):
    from utils.gcloud import lookup_tpu_ip_by_name, configure_env_for_tpu

    print("Configuring ENV variables for TPU training")
    tpu_ip = lookup_tpu_ip_by_name(tpu_name)
    configure_env_for_tpu(tpu_ip)


def device(gpu, tpu=None):
    use_cuda = torch.cuda.is_available()
    if tpu:
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    else:
        use_cuda = torch.cuda.is_available()
        return torch.device(f"cuda:{gpu}" if use_cuda else "cpu")


def dimension(dataset):
    if dataset == "mnist":
        input_shape, num_classes = (1, 28, 28), 10
    if dataset == "cifar10":
        input_shape, num_classes = (3, 32, 32), 10
    if dataset == "cifar100":
        input_shape, num_classes = (3, 32, 32), 100
    if dataset == "tiny-imagenet":
        input_shape, num_classes = (3, 64, 64), 200
    if dataset == "imagenet":
        input_shape, num_classes = (3, 224, 224), 1000
    return input_shape, num_classes


def get_transform(size, padding, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(transforms.RandomCrop(size=size, padding=padding))
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform)


def dataloader(
    dataset, batch_size, train, workers, length=None, datadir="Data", tpu=False
):
    # Dataset
    if dataset == "mnist":
        mean, std = (0.1307,), (0.3081,)
        transform = get_transform(
            size=28, padding=0, mean=mean, std=std, preprocess=False
        )
        dataset = datasets.MNIST(
            datadir, train=train, download=True, transform=transform
        )
    if dataset == "cifar10":
        mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
        transform = get_transform(
            size=32, padding=4, mean=mean, std=std, preprocess=train
        )
        dataset = datasets.CIFAR10(
            datadir, train=train, download=True, transform=transform
        )
    if dataset == "cifar100":
        mean, std = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
        transform = get_transform(
            size=32, padding=4, mean=mean, std=std, preprocess=train
        )
        dataset = datasets.CIFAR100(
            datadir, train=train, download=True, transform=transform
        )
    if dataset == "tiny-imagenet":
        mean, std = (0.480, 0.448, 0.397), (0.276, 0.269, 0.282)
        transform = get_transform(
            size=64, padding=4, mean=mean, std=std, preprocess=train
        )
        dataset = custom_datasets.TINYIMAGENET(
            datadir, train=train, download=True, transform=transform
        )
    if dataset == "imagenet":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if train:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        folder = f"{datadir}/imagenet_raw/{'train' if train else 'val'}"
        dataset = datasets.ImageFolder(folder, transform=transform)

    # Dataloader
    shuffle = train is True
    if length is not None:
        indices = torch.randperm(len(dataset))[:length]
        dataset = torch.utils.data.Subset(dataset, indices)

    sampler = None
    kwargs = {}
    if torch.cuda.is_available():
        kwargs = {"num_workers": workers, "pin_memory": True}
    elif tpu:
        import torch_xla.core.xla_model as xm

        # TODO: might want to drop last to keep batches the same size and
        # speed up computation
        kwargs = {"num_workers": workers}  # , "drop_last": True}
        if xm.xrt_world_size() > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=shuffle,
            )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False if sampler else shuffle,
        sampler=sampler,
        **kwargs,
    )

    return dataloader


def model(model_architecture, model_class):
    default_models = {
        "logistic": mlp.logistic,
        "fc": mlp.fc,
        "fc-bn": mlp.fc_bn,
        "conv": mlp.conv,
    }
    tinyimagenet_models = {
        "vgg11": tinyimagenet_vgg.vgg11,
        "vgg11-bn": tinyimagenet_vgg.vgg11_bn,
        "vgg13": tinyimagenet_vgg.vgg13,
        "vgg13-bn": tinyimagenet_vgg.vgg13_bn,
        "vgg16": tinyimagenet_vgg.vgg16,
        "vgg16-bn": tinyimagenet_vgg.vgg16_bn,
        "vgg19": tinyimagenet_vgg.vgg19,
        "vgg19-bn": tinyimagenet_vgg.vgg19_bn,
        "resnet18": tinyimagenet_resnet.resnet18,
        "resnet34": tinyimagenet_resnet.resnet34,
        "resnet50": tinyimagenet_resnet.resnet50,
        "resnet101": tinyimagenet_resnet.resnet101,
        "resnet152": tinyimagenet_resnet.resnet152,
        "wide-resnet18": tinyimagenet_resnet.wide_resnet18,
        "wide-resnet34": tinyimagenet_resnet.wide_resnet34,
        "wide-resnet50": tinyimagenet_resnet.wide_resnet50,
        "wide-resnet101": tinyimagenet_resnet.wide_resnet101,
        "wide-resnet152": tinyimagenet_resnet.wide_resnet152,
        "resnet18-nobn": tinyimagenet_resnet.resnet18_nobn,
        "resnet34-nobn": tinyimagenet_resnet.resnet34_nobn,
        "resnet50-nobn": tinyimagenet_resnet.resnet50_nobn,
        "resnet101-nobn": tinyimagenet_resnet.resnet101_nobn,
        "resnet152-nobn": tinyimagenet_resnet.resnet152_nobn,
        "wide-resnet18-nobn": tinyimagenet_resnet.wide_resnet18_nobn,
        "wide-resnet34-nobn": tinyimagenet_resnet.wide_resnet34_nobn,
        "wide-resnet50-nobn": tinyimagenet_resnet.wide_resnet50_nobn,
        "wide-resnet101-nobn": tinyimagenet_resnet.wide_resnet101_nobn,
        "wide-resnet152-nobn": tinyimagenet_resnet.wide_resnet152_nobn,
    }
    imagenet_models = {
        "vgg11": imagenet_vgg.vgg11,
        "vgg11-bn": imagenet_vgg.vgg11_bn,
        "vgg13": imagenet_vgg.vgg13,
        "vgg13-bn": imagenet_vgg.vgg13_bn,
        "vgg16": imagenet_vgg.vgg16,
        "vgg16-bn": imagenet_vgg.vgg16_bn,
        "vgg19": imagenet_vgg.vgg19,
        "vgg19-bn": imagenet_vgg.vgg19_bn,
        "resnet18": imagenet_resnet.resnet18,
        "resnet34": imagenet_resnet.resnet34,
        "resnet50": imagenet_resnet.resnet50,
        "resnet101": imagenet_resnet.resnet101,
        "resnet152": imagenet_resnet.resnet152,
        "wide-resnet50": imagenet_resnet.wide_resnet50_2,
        "wide-resnet101": imagenet_resnet.wide_resnet101_2,
    }
    models = {
        "default": default_models,
        "tinyimagenet": tinyimagenet_models,
        "imagenet": imagenet_models,
    }
    return models[model_class][model_architecture]


def optimizer(optimizer, momentum=0.0, dampening=0.0, nesterov=False, save_buffers=[]):
    optimizers = {
        "custom_sgd": (
            custom_sgd.SGD,
            {
                "momentum": momentum,
                "dampening": dampening,
                "nesterov": nesterov,
                "save_buffers": save_buffers,
            },
        ),
        "sgd": (optim.SGD, {}),
        "momentum": (
            optim.SGD,
            {"momentum": momentum, "dampening": dampening, "nesterov": nesterov},
        ),
        "adam": (optim.Adam, {}),
        "rms": (optim.RMSprop, {}),
        "lamb": (lamb.Lamb, {}),
    }
    return optimizers[optimizer]
