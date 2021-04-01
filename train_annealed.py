import json
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from utils import load
from utils import optimize
from utils import flags


def anneal_schedule(anneal_steps, args):
    # Constructs a hyperparam anneal schedule based on
    # doubling batch size at every step but keeping two quantities of interest fixed
    sch_keys = ["train_batch_size", "lr", "momentum", "save_freq"]
    for k in sch_keys:
        assert k in args.keys()

    schedule = [{k:v for k,v in args.items() if k in sch_keys}]
    for i in range(1, anneal_steps):
        these_args = schedule[i-1].copy()
        these_args["train_batch_size"] *= 2
        these_args["save_freq"] = max(1, int(these_args["save_freq"]/2))
        beta = these_args["momentum"]
        beta_hat = np.sqrt((1 + beta**2)/2)
        eta_hat = 2*these_args["lr"]*(1-beta_hat)/(1-beta)
        these_args["momentum"] = beta_hat
        these_args["lr"] = eta_hat
        schedule.append(these_args)

        # Check that we have different values for the keys of interest
        for k in sch_keys[:-1]:
            assert schedule[i-1][k] != schedule[i][k]
        # Check that the quantities of interest did not change
        invariant1 = schedule[i-1]["lr"]/(2*schedule[i-1]["train_batch_size"]*(1-schedule[i-1]["momentum"]))
        invariant2 = schedule[i]["lr"]/(2*schedule[i]["train_batch_size"]*(1-schedule[i]["momentum"]))
        assert np.allclose(invariant1, invariant2, atol=1e-8)
        invariant1 = 1/(schedule[i-1]["train_batch_size"]*(1-schedule[i-1]["momentum"]**2))
        invariant2 = 1/(schedule[i]["train_batch_size"]*(1-schedule[i]["momentum"]**2))
        assert np.allclose(invariant1, invariant2, atol=1e-8)

    return schedule


def anneal_schedule_lr(anneal_steps, args):
    # Constructs a hyperparam anneal schedule based on
    # doubling batch size at every step but keeping two quantities of interest fixed
    sch_keys = ["lr", "momentum"]
    for k in sch_keys:
        assert k in args.keys()

    schedule = [{k:v for k,v in args.items() if k in sch_keys}]
    lr_start = schedule[0]["lr"]
    lr_end = 1e-8
    mom_start = schedule[0]["momentum"]
    mom_end = 0.9999

    etas = np.linspace(lr_start, lr_end, anneal_steps)
    betas = np.linspace(mom_start, mom_end, anneal_steps)
    for i in range(1, anneal_steps):

        these_args["momentum"] = betas[i]
        these_args["lr"] = etas[i]
        schedule.append(these_args)

        # Check that we have different values for the keys of interest
        for k in sch_keys:
            assert schedule[i-1][k] != schedule[i][k]
        # Check that the quantities of interest did not change
        invariant1 = schedule[i-1]["lr"]/(2*schedule[i-1]["train_batch_size"]*(1-schedule[i-1]["momentum"]))
        invariant2 = schedule[i]["lr"]/(2*schedule[i]["train_batch_size"]*(1-schedule[i]["momentum"]))
        assert np.allclose(invariant1, invariant2, atol=1e-8)
        invariant1 = 1/(schedule[i-1]["train_batch_size"]*(1-schedule[i-1]["momentum"]**2))
        invariant2 = 1/(schedule[i]["train_batch_size"]*(1-schedule[i]["momentum"]**2))
        assert np.allclose(invariant1, invariant2, atol=1e-8)

    return schedule


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
        filename = save_path + "/hyperparameters.json"
        with open(filename, "w") as f:
            json.dump(ARGS.__dict__, f, sort_keys=True, indent=4)
        if ARGS.tpu:
            if xm.get_ordinal() == 0 and filename[0:5] == "gs://":
                from utils.gcloud import post_file_to_bucket

                post_file_to_bucket(filename)

    ## Random Seed and Device ##
    torch.manual_seed(ARGS.seed)
    device = load.device(ARGS.gpu, tpu=ARGS.tpu)

    ## Model ##
    print_fn("Creating {}-{} model.".format(ARGS.model_class, ARGS.model))
    # Data-dependent piece of the model
    input_shape, num_classes = load.dimension(ARGS.dataset)
    model = load.model(ARGS.model, ARGS.model_class)(
        input_shape=input_shape, num_classes=num_classes, pretrained=ARGS.pretrained,
        model_dir=ARGS.model_dir,
    ).to(device)

    # Construct the annealing schedule
    schedule = anneal_schedule(ARGS.anneal_steps, vars(ARGS))
    # Run one epoch of training at every step in the schedule
    epoch_offset = -1
    for k,sch_args in enumerate(schedule):
        # Set the schedule args
        ARGS.__dict__.update(sch_args)
        print_fn("Running with args: {}".format(ARGS))
        epoch_offset += ARGS.epochs*(2**k) # To prevent step number from being overwritten, given the current batch size schedule

        ## Data ##
        print_fn("Loading {} dataset.".format(ARGS.dataset))
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
        train_kwargs = {
            "batch_size": train_loader.batch_size,
            "dataset_size": len(train_loader.dataset),
            "num_batches": len(train_loader),
        }
        if ARGS.tpu:
            train_kwargs.update(
                {"xrt_world_size": xm.xrt_world_size(), "xm_ordinal": xm.get_ordinal(),}
            )

        ## Loss ##
        loss = nn.CrossEntropyLoss()
        ## Optimizer ##
        opt_class, opt_kwargs = load.optimizer(
            ARGS.optimizer, ARGS.momentum, ARGS.dampening, ARGS.nesterov, ARGS.save_buffers,
        )
        opt_kwargs.update({"lr": ARGS.lr, "weight_decay": ARGS.wd})
        optimizer = opt_class(model.parameters(), **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=ARGS.lr_drops, gamma=ARGS.lr_drop_rate
        )

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
            epoch_offset=epoch_offset,
            **train_kwargs,
        )


if __name__ == "__main__":
    parser = flags.train()
    ARGS = parser.parse_args()
    flags.validate_train(ARGS)
    if ARGS.tpu:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp

        load.configure_tpu(ARGS.tpu)

        def _mp_fn(rank, args):
            main(args)

        xmp.spawn(_mp_fn, args=(ARGS,), nprocs=None, start_method="fork")
    else:
        main(ARGS)
