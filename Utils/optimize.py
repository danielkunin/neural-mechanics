import torch
import numpy as np
from tqdm import tqdm


def checkpoint(
    model, optimizer, scheduler, epoch, curr_step, save_path, metric_dict={}
):
    print(f"Saving model checkpoint for step {curr_step}")
    save_dict = {
        "epoch": epoch,
        "step": curr_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    save_dict.update(metric_dict)
    torch.save(
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
    log_interval=10,
    save_freq=100,
    save_steps=None,
    save_path=None,
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
        if save and save_path is not None and save_steps is not None:
            if len(save_steps) > 0 and curr_step == save_steps[0]:
                save_steps.pop(0)
                checkpoint(
                    model, optimizer, scheduler, epoch, curr_step, save_path, eval_dict
                )

    return total / len(dataloader.dataset)


def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:, :1].sum().item()
            correct5 += correct[:, :5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100.0 * correct1 / len(dataloader.dataset)
    accuracy5 = 100.0 * correct5 / len(dataloader.dataset)
    if verbose:
        print(
            "Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)".format(
                average_loss, correct1, len(dataloader.dataset), accuracy1
            )
        )
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
    save_freq=100,
    save_steps=None,
    save_path=None,
):
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    metric_dict = {
        "train_loss": 0,
        "test_loss": test_loss,
        "accuracy1": accuracy1,
        "accuracy5": accuracy5,
    }
    if save:
        checkpoint(model, optimizer, scheduler, 0, 0, save_path, metric_dict)
    for epoch in tqdm(range(epochs)):
        train_loss = train(
            model,
            loss,
            optimizer,
            scheduler,
            train_loader,
            device,
            epoch,
            verbose,
            save,
            save_freq=save_freq,
            save_steps=save_steps,
            save_path=save_path,
        )
        test_loss, accuracy1, accuracy5 = eval(
            model, loss, test_loader, device, verbose
        )
        metric_dict = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "accuracy1": accuracy1,
            "accuracy5": accuracy5,
        }
        curr_step = (epoch + 1) * len(train_loader)
        if save:
            checkpoint(
                model, optimizer, scheduler, epoch, curr_step, save_path, metric_dict
            )
        scheduler.step()