import torch
import numpy as np
from tqdm import tqdm


def checkpoint(
    model, optimizer, scheduler, epoch, curr_step, save_path, metric_dict={}, tpu=False
):
    if tpu:
        import torch_xla.core.xla_model as xm

        save_lib = xm
        print_fn = xm.master_print
    else:
        save_lib = torch
        print_fn = print
    print_fn(f"Saving model checkpoint for step {curr_step}")
    save_dict = {
        "epoch": epoch,
        "step": curr_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    save_dict.update(metric_dict)
    save_lib.save(
        save_dict, f"{save_path}/ckpt/step{curr_step}.tar",
    )


# TODO: we maybe don't want to have the scheduler inside the train function
def train(
    model,
    loss,
    optimizer,
    scheduler,
    dataloader,
    device,
    epoch,
    verbose,
    save,
    save_freq,
    save_path,
    log_interval=10,
    **kwargs,
):
    batch_size = kwargs.get("batch_size")  # per core batch size
    num_batches = kwargs.get("num_batches")  #  len(dataloader)
    dataset_size = kwargs.get("dataset_size")  # len(dataloader.dataset)
    if device.type == "xla":
        import torch_xla.core.xla_model as xm

        xrt_world_size = kwargs.get("xrt_world_size")
        xm_ordinal = kwargs.get("xm_ordinal")
        tracker = xm.RateTracker()

    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        if device.type == "xla":
            step = batch_idx * xrt_world_size + xm_ordinal
            curr_step = epoch * num_batches + step
            ckpt_step = epoch * num_batches + batch_idx * xrt_world_size
        else:
            data, target = data.to(device), target.to(device)
            step = batch_idx
            curr_step = epoch * num_batches + step
            ckpt_step = curr_step

        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        if device.type == "xla":
            xm.optimizer_step(optimizer)
            tracker.add(batch_size)
        else:
            optimizer.step()
        curr_step += 1
        if verbose & (batch_idx % log_interval == 0):
            per_worker_header = ""
            if device.type == "xla":
                per_worker_header = (
                    f"[xla:{xm.get_ordinal()}, "
                    f"rate: {tracker.rate():.2f}, "
                    f"global_rate: {tracker.global_rate():.2f}]\t"
                )
            print(
                f"{per_worker_header}"
                f"Train Epoch: {epoch} "
                f"[{step*batch_size}/{dataset_size} "
                f"({100.0*batch_idx/num_batches:.0f}%)]"
                f"\tLoss: {train_loss.item():.6f}"
                f"\tStep: {curr_step}"
            )
        # TODO: this is just to be able to save at any step (even mid-epoch)
        #       it might make more sense to checkpoint only on epoch: makes
        #       for a cleaner codebase and can include test metrics
        # TODO: additionally, could integrate tfutils.DBInterface here
        eval_dict = {"train_loss": train_loss.item()}
        if save and save_path is not None and save_freq is not None:
            # TODO: think about ckpt_step and how to make it be the exact
            # checkpoint step. Multiprocessing will wait for all threads to 
            # hit this save instruction before continuing, so the ckpt_step needs to be 
            # the same for all threads. This means the checkpoint will be created then
            # one of the threads hits the save_freq, but the model itself will be
            # at that step +- num_cores
            if ckpt_step % save_freq == 0:
                checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    ckpt_step,
                    save_path,
                    tpu=(device.type == "xla"),
                )
    return total / dataset_size


def eval(model, loss, dataloader, device, verbose, **kwargs):
    if device.type == "xla":
        import torch_xla.core.xla_model as xm

        print_fn = xm.master_print
    else:
        print_fn = print

    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:, :1].sum().item()
            correct5 += correct[:, :5].sum().item()
            total_samples += data.size()[0]
    average_loss = 1.0 * total / total_samples
    accuracy1 = 100.0 * correct1 / total_samples
    accuracy5 = 100.0 * correct5 / total_samples
    if verbose:
        print_fn(
            "Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)".format(
                average_loss, correct1, total_samples, accuracy1
            )
        )
    # TODO: For tpu MP, might need to mesh_reduce the metrics?
    # if device.type == "xla":
    #     accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
    return average_loss, accuracy1, accuracy5


def train_eval_loop(
    model,
    loss,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    device,
    epochs,
    verbose,
    save,
    save_freq=None,
    save_path=None,
    **kwargs,
):
    if device.type == "xla":
        import torch_xla.distributed.parallel_loader as pl

        def loader_wrap(loader):
            # For torch-xla == 1.5, it is necessary to call at every epoch,
            # cause otherwise the iterator does not reinitialize
            # torch-xla == 1.6 introduces pl.MpDeviceLoader(test_loader, device)
            para_loader = pl.ParallelLoader(loader, [device])
            return para_loader.per_device_loader(device)

    else:

        def loader_wrap(loader):
            return loader

    test_loss, accuracy1, accuracy5 = eval(
        model, loss, loader_wrap(test_loader), device, verbose
    )
    metric_dict = {
        "train_loss": 0,
        "test_loss": test_loss,
        "accuracy1": accuracy1,
        "accuracy5": accuracy5,
    }
    if save:
        checkpoint(
            model,
            optimizer,
            scheduler,
            0,
            0,
            save_path,
            metric_dict,
            tpu=(device.type == "xla"),
        )
    for epoch in tqdm(range(epochs)):
        train_loss = train(
            model,
            loss,
            optimizer,
            scheduler,
            loader_wrap(train_loader),
            device,
            epoch,
            verbose,
            save,
            save_freq=save_freq,
            save_path=save_path,
            **kwargs,
        )
        test_loss, accuracy1, accuracy5 = eval(
            model, loss, loader_wrap(test_loader), device, verbose
        )
        metric_dict = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "accuracy1": accuracy1,
            "accuracy5": accuracy5,
        }
        curr_step = (epoch + 1) * kwargs.get("num_batches")
        if save:
            checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                curr_step,
                save_path,
                metric_dict,
                tpu=(device.type == "xla"),
            )
        scheduler.step()
