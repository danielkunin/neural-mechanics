from tqdm import tqdm
import metrics.helper as utils
import numpy as np


def compute_pos_vel(step, layers, load_kwargs, position, velocity, **kwargs):
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
            suffix="weight.grad_norm_buffer",
            group="buffers",
            **load_kwargs,
        )
    bias_buffers = utils.load_features(
            steps=[str(step)],
            suffix="bias.grad_norm_buffer",
            group="buffers",
            **load_kwargs,
        )

    for layer in layers:
        Wl_t = weights[layer][f"step_{step}"]
        bl_t = biases[layer][f"step_{step}"]
        position[layer][step] = utils.in_synapses(Wl_t ** 2, bl_t ** 2)

        g_Wl_t = weight_buffers[layer][f"step_{step}"]
        g_bl_t = bias_buffers[layer][f"step_{step}"]
        # -2lambda |\theta|^2 + \eta(|g|^2 - \lambda^2|\theta|^2)
        velocity[layer][step] = lr*utils.in_synapses(g_Wl_t, g_bl_t)
        velocity[layer][step] -= (2*wd + lr*wd**2)*position[layer][step]


def phase(model, feats_dir, steps, **kwargs):
    lr = kwargs.get("lr")
    wd = kwargs.get("wd")

    layers = [layer for layer in utils.get_layers(model) if "conv" in layer]

    load_kwargs = {
        "model": model,
        "feats_dir": feats_dir,
    }

    position = {layer: {} for layer in layers}
    velocity = {layer: {} for layer in layers}
    for i in tqdm(range(1, len(steps))):
        step = steps[i]
        compute_pos_vel(step, layers, load_kwargs, position, velocity, **kwargs)

    return {"position": position, "velocity": velocity}
