import argparse
import json
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from Utils import load
from Utils import optimize

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Neural Mechanics")
    # Training Hyperparameters
    training_args = parser.add_argument_group("training")
    training_args.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "cifar100", "tiny-imagenet", "imagenet"],
        help="dataset (default: mnist)",
    )
    training_args.add_argument(
        "--data-dir",
        type=str,
        default="Data",
        help="Directory to store the datasets to be downloaded",
    )
    training_args.add_argument(
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
            "resnet18-nobn",
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
    training_args.add_argument(
        "--model-class",
        type=str,
        default="default",
        choices=["default","tinyimagenet", "imagenet"],
        help="model class (default: default)",
    )
    training_args.add_argument(
        "--pretrained",
        type=bool,
        default=False,
        help="load pretrained weights (default: False)",
    )
    training_args.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["custom_sgd", "sgd", "momentum", "adam", "rms"],
        help="optimizer (default: sgd)",
    )
    training_args.add_argument(
        "--train-batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    training_args.add_argument(
        "--test-batch-size",
        type=int,
        default=256,
        help="input batch size for testing (default: 256)",
    )
    training_args.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="number of epochs to train (default: 0)",
    )
    training_args.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    training_args.add_argument(
        "--lr-drops",
        type=int,
        nargs="*",
        default=[],
        help="list of learning rate drops (default: [])",
    )
    training_args.add_argument(
        "--lr-drop-rate",
        type=float,
        default=0.1,
        help="multiplicative factor of learning rate drop (default: 0.1)",
    )
    training_args.add_argument(
        "--weight-decay", type=float, default=0.0, help="weight decay (default: 0.0)"
    )

    ## Experiment and Checkpointing Hyperparameters ##
    experiment_args = parser.add_argument_group("experiment")
    experiment_args.add_argument(
        "--experiment", type=str, default="", help='name used to save results (default: "")'
    )
    experiment_args.add_argument(
        "--expid", type=str, default="", help='name used to save results (default: "")'
    )
    experiment_args.add_argument(
        "--save-dir",
        type=str,
        default="Results",
        help='Directory to save checkpoints and features (default: "Results")',
    )
    experiment_args.add_argument(
        "--save-freq",
        type=int,
        default=100,
        help="Frequency (in batches) to save model checkpoints at",
    )
    experiment_args.add_argument(
        "--steps-file",
        type=str,
        default=None,
        help="File to read the checkpoints steps from. Will override step_freq",
    )
    experiment_args.add_argument(
        "--gpu", type=int, default="0", help="number of GPU device to use (default: 0)"
    )
    experiment_args.add_argument(
        "--workers",
        type=int,
        default="4",
        help="number of data loading workers (default: 4)",
    )
    experiment_args.add_argument("--no-cuda", action="store_true", help="disables CUDA training")
    experiment_args.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    experiment_args.add_argument(
        "--verbose",
        action="store_true",
        help="print statistics during training and testing",
    )

    args = parser.parse_args()

    ## Construct Result Directory ##
    if args.expid == "":
        print("WARNING: this experiment is not being saved.")
        setattr(args, "save", False)
        exp_path = None
    else:
        exp_path = f"{args.save_dir}/{args.experiment}/{args.expid}"
        setattr(args, "save", True)
        try:
            os.makedirs(exp_path)
            os.makedirs(f"{exp_path}/ckpt")
        except FileExistsError:
            val = ""
            while val not in ["yes", "no"]:
                val = input(
                    f"Experiment '{args.experiment}' with expid '{args.expid}' "
                    "exists.  Overwrite (yes/no)? "
                )
            if val == "no":
                quit()
            else:
                shutil.rmtree(exp_path)
                os.makedirs(exp_path)
                os.makedirs(f"{exp_path}/ckpt")

    ## Save Args ##
    if args.save:
        with open(exp_path + "/args.json", "w") as f:
            json.dump(args.__dict__, f, sort_keys=True, indent=4)

    
    ## Run Experiment ##

    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)

    ## Data ##
    print("Loading {} dataset.".format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset)
    train_loader = load.dataloader(
        dataset=args.dataset,
        batch_size=args.train_batch_size,
        train=True,
        workers=args.workers,
        datadir=args.data_dir,
    )
    test_loader = load.dataloader(
        dataset=args.dataset,
        batch_size=args.test_batch_size,
        train=False,
        workers=args.workers,
        datadir=args.data_dir,
    )

    ## Model, Loss, Optimizer ##
    print("Creating {}-{} model.".format(args.model_class, args.model))
    model = load.model(args.model, args.model_class)(
        input_shape=input_shape,
        num_classes=num_classes,
        pretrained=args.pretrained,
    ).to(device)
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        **opt_kwargs,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate
    )

    ## Checkpointing setup ##
    if args.steps_file is not None:
        save_steps = load.save_steps_file(args.steps_file)
        steps_per_epoch = len(train_loader)
        max_epochs = int(save_steps[-1] / steps_per_epoch)
        print(f"Overriding train epochs to last step in file ")
        setattr(args, "epochs", max_epochs)
    else:
        save_steps = None

    ## Train ##
    print("Training for {} epochs.".format(args.epochs))
    optimize.train_eval_loop(
        model,
        loss,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
        device,
        args.epochs,
        args.verbose,
        args.save,
        save_steps=save_steps,
        save_freq=args.save_freq,
        path=exp_path,
    )

