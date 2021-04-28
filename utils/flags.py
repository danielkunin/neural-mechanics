import argparse


def str_list(x):
    return x.split(",")


def default():
    parser = argparse.ArgumentParser(description="Neural Mechanics")
    parser.add_argument(
        "--experiment",
        type=str,
        default="",
        help='name used to save results (default: "")',
    )
    parser.add_argument(
        "--expid", type=str, default="", help='name used to save results (default: "")'
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help='Directory to save checkpoints and features (default: "results")',
    )
    parser.add_argument(
        "--gpu", type=int, default="0", help="number of GPU device to use (default: 0)"
    )
    parser.add_argument(
        "--tpu", type=str, default=None, help="Name of the TPU device to use",
    )
    parser.add_argument(
        "--overwrite", dest="overwrite", action="store_true", default=False
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
        choices=["default", "tinyimagenet", "imagenet"],
        help="model class (default: default)",
    )
    train_args.add_argument(
        "--pretrained",
        type=bool,
        default=False,
        help="load pretrained weights (default: False)",
    )
    train_args.add_argument(
        "--model-dir",
        type=str,
        default="pretrained_models",
        help="Directory for pretrained models. "
             "Save pretrained models to use here. "
             "Downloaded models will be stored here.",
    )
    train_args.add_argument(
        "--restore-path",
        type=str,
        default=None,
        help="Path to a checkpoint to restore a model from.",
    )
    train_args.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["mse", "ce",],
        help="loss funcion (default: ce)",
    )
    train_args.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["custom_sgd", "sgd", "momentum", "adam", "rms", "lamb",],
        help="optimizer (default: sgd)",
    )
    train_args.add_argument(
        "--save-buffers",
        type=str_list,
        default=[],
        help="comma separated list of which buffers to save in custom optimizer (default: [])",
    )
    train_args.add_argument(
        "--train-batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64), per core in TPU setting",
    )
    train_args.add_argument(
        "--test-batch-size",
        type=int,
        default=256,
        help="input batch size for testing (default: 256), per core in TPU setting",
    )
    train_args.add_argument(
        "--epochs", type=int, default=0, help="number of epochs to train (default: 0)",
    )
    train_args.add_argument(
        "--anneal-steps", type=int, default=1,
        help="number times to anneal hyperparameters(default: 1)",
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
        "--momentum", type=float, default=0.0, help="momentum parameter (default: 0.0)"
    )
    train_args.add_argument(
        "--dampening",
        type=float,
        default=0.0,
        help="dampening parameter (default: 0.0)",
    )
    train_args.add_argument(
        "--nesterov",
        type=bool,
        default=False,
        help="nesterov momentum (default: False)",
    )
    train_args.add_argument(
        "--workers",
        type=int,
        default="4",
        help="number of data loading workers (default: 4)",
    )
    train_args.add_argument(
        "--seed", type=int, default=1, help="random seed (default: 1)"
    )
    train_args.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Print statistics during training and testing. Use -vv for higher verbosity.",
    )
    train_args.add_argument(
        "--save-freq",
        type=int,
        default=None,
        help="Frequency (in batches) to save model checkpoints at",
    )
    train_args.add_argument(
        "--save-begin-epoch",
        type=float,
        default=0,
        help="Epoch at which to begin saving every save-freq steps",
    )
    # Hessian metrics
    train_args.add_argument(
        "--eigenvector",
        type=bool,
        default=False,
        help="Save Hessian eigenvectors (default: False)",
    )
    train_args.add_argument(
        "--lanczos",
        type=bool,
        default=False,
        help="Compute Hessian eigenvectors using Lanczos (default: False)",
    )
    train_args.add_argument(
        "--eigen-dims",
        type=int,
        default=1,
        help="Number of top eigenvectors and values to compute",
    )
    train_args.add_argument(
        "--eigen-batch-size",
        type=int,
        default=256,
        help="Number of top eigenvectors and values to compute",
    )
    train_args.add_argument(
        "--eigen-data-length",
        type=int,
        default=None,
        help="Number of examples to use to compute the Hessian-vector products."
             "Must be between 1 and the size of the dataset",
    )
    train_args.add_argument(
        "--power-iters",
        type=int,
        default=5,
        help="Number of iterations for the eigenvector computation",
    )
    train_args.add_argument(
        "--hessian",
        type=bool,
        default=False,
        help="Save full Hessian (default: False)",
    )
    return parser


def validate_train(parsed_args):
    for m in parsed_args.save_buffers:
        assert (
            m in ["sgd", "mom", "grad", "grad_norm"],
            "--save-buffers must be a comma separated list of these options: sgd,mom,grad,grad_norm",
        )


def extract():
    parser = default()
    return parser


def cache():
    parser = default()
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="cache and image file suffix",
        required=False,
    )
    parser.add_argument(
        "--metrics",
        type=str_list,
        default=[],
        help="comma separated list of which metrics to compute and cache. Caches all if not specified (default: [])",
    )
    return parser
