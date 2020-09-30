import torch
import numpy as np
from tqdm import tqdm

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl


def checkpoint(
    model, optimizer, scheduler, epoch, curr_step, save_path, metric_dict={}, tpu=False
):
    if tpu:
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
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        curr_step = epoch * len(dataloader) + batch_idx
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        curr_step += 1
        if verbose & (batch_idx % log_interval == 0):
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Step: {}".format(
                    epoch,
                    batch_idx * len(data),
                    len(dataloader.dataset),
                    100.0 * batch_idx / len(dataloader),
                    train_loss.item(),
                    curr_step,
                )
            )
        # TODO: this is just to be able to save at any step (even mid-epoch)
        #       it might make more sense to checkpoint only on epoch: makes
        #       for a cleaner codebase and can include test metrics
        # TODO: additionally, could integrate tfutils.DBInterface here
        eval_dict = {"train_loss": train_loss.item()}
        if save and save_path is not None and save_freq is not None:
            if curr_step % save_freq == 0:
                checkpoint(model, optimizer, scheduler, epoch, curr_step, save_path)

    return total / len(dataloader.dataset)


def tpu_train(
    model,
    loss,
    optimizer,
    scheduler,
    dataloader,
    device,
    epoch,
    verbose,
    log_interval=10,
    save_freq=100,
    save_steps=None,
    save_path=None,
    **kwargs,
):
    batch_size = kwargs.get("batch_size")
    num_batches = kwargs.get("num_batches")
    dataset_size = kwargs.get("dataset_size")
    xrt_world_size = 8 # TODO: pass from wkards
    xm_orginal = 0 # TODO: pass from kwards: too many calls baffle it
    tracker = xm.RateTracker()
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        # data, target = data.to(device), target.to(device)
        step = batch_idx*xrt_world_size #+ xm.get_ordinal()
        #step = batch_idx
        curr_step = epoch * num_batches + step
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        xm.optimizer_step(optimizer)
        tracker.add(batch_size)
        if verbose & (batch_idx % log_interval == 0):
            print(
                f"[xla:{xm.get_ordinal()}, rate: {tracker.rate():.2f}, global_rate: {tracker.global_rate():.2f}] "
                f"\tTrain Epoch: {epoch} "
                f"[{step*batch_size}/{dataset_size} "
                f"({100.0*batch_idx/num_batches:.0f}%)]"
                f"\tLoss: {train_loss.item():.6f}"
                f"\tStep: {curr_step}"
                f"\tData size: {data.size(0)}"
            )
        # TODO: this is just to be able to save at any step (even mid-epoch)
        #       it might make more sense to checkpoint only on epoch: makes
        #       for a cleaner codebase and can include test metrics
        # TODO: additionally, could integrate tfutils.DBInterface here
        ##eval_dict = {"train_loss": train_loss.item()}
        eval_dict = {}
        if save_path is not None and save_freq is not None:
            if curr_step % save_freq == 0:
                checkpoint(
                    model, optimizer, scheduler, epoch, curr_step, save_path, tpu=True
                )
        if save_path is not None and save_steps is not None:
            if len(save_steps) > 0 and curr_step == save_steps[0]:
                save_steps.pop(0)
                checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    curr_step,
                    save_path,
                    eval_dict,
                    tpu=True,
                )

    return total / dataset_size


def eval(model, loss, dataloader, device, verbose, **kwargs):
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
        print(
            "Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)".format(
                average_loss, correct1, total_samples, accuracy1
            )
        )
    return average_loss, accuracy1, accuracy5


def tpu_eval(model, loss, dataloader, device, verbose, **kwargs):
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
        xm.master_print(
            "Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)".format(
                average_loss, correct1, total_samples, accuracy1
            )
        )
    # TODO: For tpu MP, might need to mesh_reduce the metrics?
    # accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
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
        xm.master_print("Using TPU train/eval functions")
        train_fn = tpu_train
        eval_fn = tpu_eval
        # It is necessary to wrap at every epoch, cause otherwise the iterator does nor teinitialize
        def loader_wrap(loader):
            para_loader = pl.ParallelLoader(loader, [device])
            return para_loader.per_device_loader(device)
    
    else:
        train_fn = train
        eval_fn = eval

        def loader_wrap(loader):
            return loader

    test_loss, accuracy1, accuracy5 = eval_fn(
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
            tpu=device.type == "xla",
        )
    for epoch in tqdm(range(epochs)):
        train_loss = train_fn(
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
        test_loss, accuracy1, accuracy5 = eval_fn(
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
                tpu=device.type == "xla",
            )
        scheduler.step()
