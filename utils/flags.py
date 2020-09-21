import argparse

def default():
    parser = argparse.ArgumentParser(description="Neural Mechanics")
    parser.add_argument(
        "--experiment", type=str, default="", help='name used to save results (default: "")'
    )
    parser.add_argument(
        "--expid", type=str, default="", help='name used to save results (default: "")'
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help='Directory to save checkpoints and features (default: "Results")',
    )
    parser.add_argument(
        "--gpu", type=int, default="0", help="number of GPU device to use (default: 0)"
    )
    parser.add_argument(
        "--overwrite", 
        dest="overwrite", 
        action="store_true",
        default=False
    )
    return parser


def train():
    parser = default()
    train_args = parser.add_argument_group("train")
    train_args.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "cifar100", "tiny-imagenet", "imagenet"],
        help="dataset (default: mnist)",
    )
    train_args.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store the datasets to be downloaded",
    )
    train_args.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=[
            "logistic",
            "fc",
            "fc-bn",
            "conv",
            "vgg11",
            "vgg11-bn",
            "vgg13",
            "vgg13-bn",
            "vgg16",
            "vgg16-bn",
            "vgg19",
            "vgg19-bn",
            "resnet18",
            "resnet20",
            "resnet32",
            "resnet34",
            "resnet44",
            "resnet50",
            "resnet56",
            "resnet101",
            "resnet110",
            "resnet110",
            "resnet152",
            "resnet1202",
            "wide-resnet18",
            "wide-resnet20",
            "wide-resnet32",
            "wide-resnet34",
            "wide-resnet44",
            "wide-resnet50",
            "wide-resnet56",
            "wide-resnet101",
            "wide-resnet110",
            "wide-resnet110",
            "wide-resnet152",
            "wide-resnet1202",
        ],
        help="model architecture (default: logistic)",
    )
    train_args.add_argument(
        "--model-class",
        type=str,
        default="default",
        choices=["default","tinyimagenet", "imagenet"],
        help="model class (default: default)",
    )
    train_args.add_argument(
        "--pretrained",
        type=bool,
        default=False,
        help="load pretrained weights (default: False)",
    )
    train_args.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["custom_sgd", "sgd", "momentum", "adam", "rms"],
        help="optimizer (default: sgd)",
    )
    train_args.add_argument(
        "--train-batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    train_args.add_argument(
        "--test-batch-size",
        type=int,
        default=256,
        help="input batch size for testing (default: 256)",
    )
    train_args.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="number of epochs to train (default: 0)",
    )
    train_args.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    train_args.add_argument(
        "--lr-drops",
        type=int,
        nargs="*",
        default=[],
        help="list of learning rate drops (default: [])",
    )
    train_args.add_argument(
        "--lr-drop-rate",
        type=float,
        default=0.1,
        help="multiplicative factor of learning rate drop (default: 0.1)",
    )
    train_args.add_argument(
        "--wd", type=float, default=0.0, help="weight decay (default: 0.0)"
    )
    train_args.add_argument(
        "--workers",
        type=int,
        default="4",
        help="number of data loading workers (default: 4)",
    )
    train_args.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    train_args.add_argument(
        "--verbose",
        action="store_true",
        help="print statistics during training and testing",
    )
    train_args.add_argument(
        "--save-freq",
        type=int,
        default=100,
        help="Frequency (in batches) to save model checkpoints at",
    )
    return parser


def extract():
    parser = default()
    return parser