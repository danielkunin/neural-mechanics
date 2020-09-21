import json
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from utils import load
from utils import optimize
from utils import flags


def main():
    ## Construct Result Directory ##
    if ARGS.expid == "":
        print("WARNING: this experiment is not being saved.")
        setattr(ARGS, "save", False)
        save_path = None
    else:
        setattr(ARGS, "save", True)
        save_path = f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}"
        try:
            os.makedirs(save_path)
            os.makedirs(f"{save_path}/ckpt")
        except FileExistsError:
            if not ARGS.overwrite:
                print(
                    "Feature directory exists and no-overwrite specified. Rerun with --overwrite"
                )
                quit()
            shutil.rmtree(save_path)
            os.makedirs(save_path)
            os.makedirs(f"{save_path}/ckpt")

    ## Save Args ##
    if ARGS.save:
        with open(save_path + "/hyperparameters.json", "w") as f:
            json.dump(ARGS.__dict__, f, sort_keys=True, indent=4)

    ## Random Seed and Device ##
    torch.manual_seed(ARGS.seed)
    device = load.device(ARGS.gpu)

    ## Data ##
    print("Loading {} dataset.".format(ARGS.dataset))
    input_shape, num_classes = load.dimension(ARGS.dataset)
    train_loader = load.dataloader(
        dataset=ARGS.dataset,
        batch_size=ARGS.train_batch_size,
        train=True,
        workers=ARGS.workers,
        datadir=ARGS.data_dir,
    )
    test_loader = load.dataloader(
        dataset=ARGS.dataset,
        batch_size=ARGS.test_batch_size,
        train=False,
        workers=ARGS.workers,
        datadir=ARGS.data_dir,
    )

    ## Model, Loss, Optimizer ##
    print("Creating {}-{} model.".format(ARGS.model_class, ARGS.model))
    model = load.model(ARGS.model, ARGS.model_class)(
        input_shape=input_shape,
        num_classes=num_classes,
        pretrained=ARGS.pretrained,
    ).to(device)
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(ARGS.optimizer)
    optimizer = opt_class(
        model.parameters(),
        lr=ARGS.lr,
        weight_decay=ARGS.wd,
        **opt_kwargs,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=ARGS.lr_drops, gamma=ARGS.lr_drop_rate
    )

    ## Train ##
    print("Training for {} epochs.".format(ARGS.epochs))
    optimize.train_eval_loop(
        model,
        loss,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
        device,
        ARGS.epochs,
        ARGS.verbose,
        ARGS.save,
        save_freq=ARGS.save_freq,
        save_path=save_path,
    )


if __name__ == "__main__":
    parser = flags.train()
    ARGS = parser.parse_args()
    main()