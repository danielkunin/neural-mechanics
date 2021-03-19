import torch
import numpy as np
from tqdm import tqdm


def checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    curr_step,
    save_path,
    verbose,
    metric_dict={},
    tpu=False,
):
    save_lib = torch
    print_fn = print
    if tpu:
        import torch_xla.core.xla_model as xm

        save_lib = xm
        print_fn = xm.master_print

    if verbose:
        print_fn(f"Saving model checkpoint for step {curr_step}")
    save_dict = {
        "epoch": epoch,
        "step": curr_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    save_dict.update(metric_dict)
    filename = f"{save_path}/ckpt/step{curr_step}.tar"
    save_lib.save(
        save_dict, filename,
    )
    if tpu:
        if xm.get_ordinal() == 0 and filename[0:5] == "gs://":
            from utils.gcloud import post_file_to_bucket

            post_file_to_bucket(filename, verbose)


def online_metric_fn(epoch_weights):
    # Do PCA stuff
    metric_dict = {}
    print("got some stuff")
    print(epoch_weights.shape)
    from sklearn.decomposition import PCA

    def filter_outliers(projected_data, threshold):
        norms = np.linalg.norm(projected_data, axis=1)
        outlier_idx = np.where(norms > threshold)[0]
        keep_idx = np.where(norms <= threshold)[0]
        return outlier_idx, keep_idx


    def fit_pca(fit_data, project_data=None, whiten=False):
        if project_data is None:
            project_data = fit_data
        pca_mod = PCA(whiten=whiten, n_components=10)
        pca_mod.fit(fit_data)
        projected = pca_mod.transform(project_data)
        return projected, pca_mod

    fit_data = epoch_weights.detach().cpu().numpy()
    project_data = None
    projected, pca_mod = fit_pca(fit_data, project_data, whiten)
    projected_fit_data = pca_mod.transform(fit_data)

    o_idx, k_idx = filter_outliers(projected_fit_data, outlier_thresh)
    while len(o_idx) > 0:
        print(o_idx)
        print(f"Refitting without {len(o_idx)} outliers")
        fit_data = fit_data[k_idx]
        projected, pca_mod = fit_pca(fit_data, project_data, whiten)
        projected_fit_data = pca_mod.transform(fit_data)

        o_idx, k_idx = filter_outliers(projected_fit_data, outlier_thresh)

    # Save the first 10 components
    metric_dict["components"] = pca.components_
    metric_dict["component_ev"] = pca_mod.explained_variance_ratio_

    return metric_dict


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

    print_fn = print
    if device.type == "xla":
        import torch_xla.core.xla_model as xm

        xrt_world_size = kwargs.get("xrt_world_size")
        xm_ordinal = kwargs.get("xm_ordinal")
        tracker = xm.RateTracker()
        if verbose <= 1:
            print_fn = xm.master_print

    model.train()
    total_loss = 0
    total_samples = 0
    train_metrics = {}
    ############ Online metric function buffer
    # This is for vgg16-bn model, tinyimagenet
    layer = 3
    weight_dim = model.state_dict()[f"features.{layer}.weight"].flatten().shape[0]
    epoch_weights = torch.empty(0, weight_dim)
    ############
    for batch_idx, (data, target) in enumerate(dataloader):
        if device.type != "xla":
            data, target = data.to(device), target.to(device)
        curr_step = epoch * num_batches + batch_idx

        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total_loss += train_loss.item() * data.size(0)
        total_samples += data.size(0)
        train_loss.backward()
        if device.type == "xla":
            xm.optimizer_step(optimizer)
            tracker.add(batch_size)
        else:
            optimizer.step()
        curr_step += 1
        if verbose and (batch_idx % log_interval == 0):
            examples_seen = batch_idx * batch_size
            per_worker_header = ""
            if device.type == "xla" and verbose >= 2:
                per_worker_header = (
                    f"[xla:{xm_ordinal}, "
                    f"rate: {tracker.rate():.2f}, "
                    f"global_rate: {tracker.global_rate():.2f}]\t"
                )
                examples_seen *= xrt_world_size
                examples_seen += xm_ordinal * batch_size
            print_fn(
                f"{per_worker_header}"
                f"Train Epoch: {epoch} "
                f"[{examples_seen}/{dataset_size} "
                f"({100.0*batch_idx/num_batches:.0f}%)]"
                f"\tLoss: {train_loss.item():.6f}"
                f"\tStep: {curr_step}"
            )
        # TODO: this is just to be able to save at any step (even mid-epoch)
        #       it might make more sense to checkpoint only on epoch: makes
        #       for a cleaner codebase and can include test metrics
        # TODO: additionally, could integrate tfutils.DBInterface here
        if save and save_path is not None and save_freq is not None:
            if curr_step % save_freq == 0:
                checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    curr_step,
                    save_path,
                    verbose,
                    tpu=(device.type == "xla"),
                )
        ############ Online metric function buffer update
        # Ignoring biases for now
        weights = model.state_dict()[f"features.{layer}.weight"].flatten()
        epoch_weights = torch.cat((epoch_weights, weights.unsqueeze(0)), dim=0)
        ############

    average_loss = 1.0 * total_loss / total_samples
    if device.type == "xla":
        average_loss = xm.mesh_reduce("train_average_loss", average_loss, np.mean)
    ########### Online metric function
    if (device.type == "xla" and xm_ordinal == 0) or device.type != "xla":
        train_metrics.update(online_metric_fn(epoch_weights))

    return average_loss, train_metrics


def eval(model, loss, dataloader, device, verbose, epoch, **kwargs):
    print_fn = print
    if device.type == "xla":
        import torch_xla.core.xla_model as xm

        print_fn = xm.master_print

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
    print_fn(
        f"Epoch {epoch} evaluation: Average loss: {average_loss:.4f}, "
        f"Top 1 Accuracy: {correct1}/{total_samples} ({accuracy1:.2f}%)"
    )

    if device.type == "xla":
        average_loss = xm.mesh_reduce("test_average_loss", average_loss, np.mean)
        accuracy1 = xm.mesh_reduce("test_accuracy1", accuracy1, np.mean)
        accuracy5 = xm.mesh_reduce("test_accuracy5", accuracy5, np.mean)
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
    print_fn = print
    if device.type == "xla":
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.core.xla_model as xm

        print_fn = xm.master_print
        train_loader = pl.MpDeviceLoader(train_loader, device)
        test_loader = pl.MpDeviceLoader(test_loader, device)

    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose, 0)
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
            verbose,
            metric_dict,
            tpu=(device.type == "xla"),
        )
    for epoch in tqdm(range(epochs)):
        train_loss, train_metrics = train(
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
            save_path=save_path,
            **kwargs,
        )
        test_loss, accuracy1, accuracy5 = eval(
            model, loss, test_loader, device, verbose, epoch + 1
        )
        metric_dict = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "accuracy1": accuracy1,
            "accuracy5": accuracy5,
            "train_metrics": train_metrics,
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
                verbose,
                metric_dict,
                tpu=(device.type == "xla"),
            )
        scheduler.step()
    print_fn(
        f"Final performance: "
        f"\tTrain Loss: {train_loss:.4f}"
        f"\tTest Loss: {test_loss:.4f}"
        f"\tAccuracy: {accuracy1:.2f}%"
    )
