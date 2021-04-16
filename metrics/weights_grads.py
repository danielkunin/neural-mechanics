from tqdm import tqdm
import metrics.helper as utils
import numpy as np


def extract_weights_and_grads(step, layers, load_kwargs, weights_and_grads, **kwargs):
    lr = kwargs.get("lr")
    wd = kwargs.get("wd")

    weights = utils.load_features(
            steps=[str(step)],
            suffix="weight",
            group="params",
            **load_kwargs,
        )
    biases = utils.load_features(
            steps=[str(step)],
            suffix="bias",
            group="params",
            **load_kwargs,
        )

    weight_buffers = utils.load_features(
            steps=[str(step)],
            suffix="weight.grad_buffer",
            group="buffers",
            **load_kwargs,
        )
    bias_buffers = utils.load_features(
            steps=[str(step)],
            suffix="bias.grad_buffer",
            group="buffers",
            **load_kwargs,
        )

    for layer in layers:
        Wl_t = weights[layer][f"step_{step}"]
        bl_t = biases[layer][f"step_{step}"]
        weights_and_grads[layer]["weight"].append(
            np.concatenate((Wl_t.flatten(), bl_t.flatten()))
        )

        g_Wl_t = weight_buffers[layer][f"step_{step}"]
        g_bl_t = bias_buffers[layer][f"step_{step}"]
        weights_and_grads[layer]["grad"].append(
            np.concatenate((g_Wl_t.flatten(), g_bl_t.flatten()))
        )


def weights_grads(model, feats_dir, steps, **kwargs):
    lr = kwargs.get("lr")
    wd = kwargs.get("wd")

    layers = [layer for layer in utils.get_layers(model) if "conv" in layer]
    load_kwargs = {
        "model": model,
        "feats_dir": feats_dir,
    }

    weights_and_grads = {layer: {"weight":[],"grad":[]} for layer in layers}
    steps = np.unique(steps)
    steps.sort()
    for i in tqdm(range(1, len(steps))):
        step = steps[i]
        extract_weights_and_grads(step, layers, load_kwargs, weights_and_grads, **kwargs)

    print("Allocating numpy arrays")
    weights_and_grads["steps"] = steps[1:]
    for layer in layers:
        weights_and_grads[layer]["weight"] = np.array(weights_and_grads[layer]["weight"])
        weights_and_grads[layer]["grad"] = np.array(weights_and_grads[layer]["grad"])

    return weights_and_grads

def weights_grads_full(model, feats_dir, steps, **kwargs):
    lr = kwargs.get("lr")
    wd = kwargs.get("wd")

    layers = [layer for layer in utils.get_layers(model)]# if "conv" in layer]
    load_kwargs = {
        "model": model,
        "feats_dir": feats_dir,
    }

    weights_and_grads = {layer: {"weight":[],"grad":[]} for layer in layers}
    steps = np.unique(steps)
    steps.sort()
    for i in tqdm(range(1, len(steps))):
        step = steps[i]
        extract_weights_and_grads(step, layers, load_kwargs, weights_and_grads, **kwargs)

    print("Allocating numpy arrays")
    all_weights = []
    all_grads = []
    for layer in layers:
        all_weights.append(np.array(weights_and_grads[layer]["weight"]))
        all_grads.append(np.array(weights_and_grads[layer]["grad"]))

    all_weights_and_grads = {
        "weights": np.concatenate(all_weights, axis=1),
        "grads": np.concatenate(all_grads, axis=1),
        "steps": steps[1:],
    }

    return all_weights_and_grads
