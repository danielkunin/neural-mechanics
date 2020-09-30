import json
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from utils import load
from utils import optimize
from utils import flags

# TPU
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


def main(ARGS):
    if ARGS.tpu:
        print_fn = xm.master_print
    else:
        print_fn = print

    ## Construct Result Directory ##
    if ARGS.expid == "":
        print_fn("WARNING: this experiment is not being saved.")
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
                print_fn(
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
    device = load.device(ARGS.gpu, tpu=ARGS.tpu)

    ## Data ##
    print_fn("Loading {} dataset.".format(ARGS.dataset))
    input_shape, num_classes = load.dimension(ARGS.dataset)
    train_loader = load.dataloader(
        dataset=ARGS.dataset,
        batch_size=ARGS.train_batch_size,
        train=True,
        workers=ARGS.workers,
        datadir=ARGS.data_dir,
        tpu=ARGS.tpu,
    )
    test_loader = load.dataloader(
        dataset=ARGS.dataset,
        batch_size=ARGS.test_batch_size,
        train=False,
        workers=ARGS.workers,
        datadir=ARGS.data_dir,
        tpu=ARGS.tpu,
    )

    ## Model, Loss, Optimizer ##
    print_fn("Creating {}-{} model.".format(ARGS.model_class, ARGS.model))
    model = load.model(ARGS.model, ARGS.model_class)(
        input_shape=input_shape, num_classes=num_classes, pretrained=ARGS.pretrained,
    ).to(device)

    if ARGS.tpu:
        # Model wrapper:
        # For the MP approach: nothing
        # For the DP approach: (this would also change how data is fed in train loop)
        # model_parallel = dp.DataParallel(model, device_ids=devices)

        # LR Rescale
        ARGS.lr *= xm.xrt_world_size()

    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(ARGS.optimizer)
    optimizer = opt_class(
        model.parameters(), lr=ARGS.lr, weight_decay=ARGS.wd, **opt_kwargs,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=ARGS.lr_drops, gamma=ARGS.lr_drop_rate
    )

    if ARGS.tpu:
        # For torch_xla == 1.5
        train_kwargs = {
            "batch_size": train_loader.batch_size,
            "dataset_size": len(train_loader.dataset),
            "num_batches": len(train_loader),
            } # TODO: pass ordinal and world size here

    ## Train ##
    print_fn("Training for {} epochs.".format(ARGS.epochs))
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
        **train_kwargs,
    )


if __name__ == "__main__":
    parser = flags.train()
    ARGS = parser.parse_args()
    if ARGS.tpu:
        # TODO: check: function might need to take a "rank" argument?
        tpu_cores = 8

        def _mp_fn(rank, args):
            main(args)

        xmp.spawn(_mp_fn, args=(ARGS,), nprocs=tpu_cores, start_method="fork")
    else:
        main(ARGS)
