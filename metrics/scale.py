from tqdm import tqdm
import metrics.helper as utils
import numpy as np


def compute_empirical(step, layers, load_kwargs, empirical):
    weights = utils.load_features(steps=[str(step)], suffix="weight", **load_kwargs,)
    biases = utils.load_features(steps=[str(step)], suffix="bias", **load_kwargs,)
    for layer in layers:
        Wl_t = weights[layer][f"step_{step}"]
        bl_t = biases[layer][f"step_{step}"]
        empirical[layer][step] = utils.in_synapses(Wl_t ** 2, bl_t ** 2)


def compute_theoretical(
    step, layers, load_kwargs, theoretical, i, step_0, lr, wd, W_0, b_0,
):
    t = lr * step
    if i > 0:
        weight_buffers = utils.load_features(
            steps=[str(step)], suffix="weight.integral_buffer", **load_kwargs,
        )
        bias_buffers = utils.load_features(
            steps=[str(step)], suffix="bias.integral_buffer", **load_kwargs,
        )

    for layer in layers:
        Wl_0 = W_0[layer][f"step_{step_0}"]
        bl_0 = b_0[layer][f"step_{step_0}"]
        theoretical[layer][step] = np.exp(-2 * wd * t) * utils.in_synapses(
            Wl_0 ** 2, bl_0 ** 2
        )
        if i > 0:
            g_Wl_t = weight_buffers[layer][f"step_{step}"]
            g_bl_t = bias_buffers[layer][f"step_{step}"]
            theoretical[layer][step] += (
                (lr ** 2) * np.exp(-2 * wd * t) * utils.in_synapses(g_Wl_t, g_bl_t)
            )


def compute_theoretical_momentum(
    step,
    layers,
    load_kwargs,
    theoretical,
    i,
    step_0,
    lr,
    wd,
    momentum,
    dampening,
    omega,
    gamma,
    W_0,
    b_0,
):
    t = lr * (1 - dampening) * step

    if i > 0:
        weight_buffers_1 = utils.load_features(
            steps=[str(step)], suffix="weight.integral_buffer_1", **load_kwargs,
        )
        bias_buffers_1 = utils.load_features(
            steps=[str(step)], suffix="bias.integral_buffer_1", **load_kwargs,
        )
        weight_buffers_2 = utils.load_features(
            steps=[str(step)], suffix="weight.integral_buffer_2", **load_kwargs,
        )
        bias_buffers_2 = utils.load_features(
            steps=[str(step)], suffix="bias.integral_buffer_2", **load_kwargs,
        )

    for layer in layers:
        Wl_0 = W_0[layer][f"step_{step_0}"]
        bl_0 = b_0[layer][f"step_{step_0}"]
        if gamma < omega:
            cos = np.cos(np.sqrt(omega ** 2 - gamma ** 2) * t)
            sin = np.sin(np.sqrt(omega ** 2 - gamma ** 2) * t)
            scale = np.exp(-gamma * t) * (
                cos + gamma / np.sqrt(omega ** 2 - gamma ** 2) * sin
            )
        elif gamma == omega:
            scale = np.exp(-gamma * t) * (1 + gamma * t)
        else:
            alpha_p = -gamma + np.sqrt(gamma ** 2 - omega ** 2)
            alpha_m = -gamma - np.sqrt(gamma ** 2 - omega ** 2)
            numer = alpha_p * np.exp(alpha_m * t) - alpha_m * np.exp(alpha_p * t)
            denom = alpha_p - alpha_m
            scale = numer / denom
        theoretical[layer][step] = scale * utils.in_synapses(
            Wl_0 ** 2, bl_0 ** 2, dtype=np.float128
        )
        if i > 0:
            g_Wl_t_1 = weight_buffers_1[layer][f"step_{step}"]
            g_bl_t_1 = bias_buffers_1[layer][f"step_{step}"]
            g_Wl_t_2 = weight_buffers_2[layer][f"step_{step}"]
            g_bl_t_2 = bias_buffers_2[layer][f"step_{step}"]

            if gamma < omega:
                sqrt = np.sqrt(omega ** 2 - gamma ** 2)
                scale_1 = np.exp(-gamma * t) * np.sin(sqrt * t) / sqrt
                scale_2 = -np.exp(-gamma * t) * np.cos(sqrt * t) / sqrt

            elif gamma == omega:
                scale_1 = np.exp(-gamma * t) * t
                scale_2 = -np.exp(-gamma * t)

            else:
                sqrt = np.sqrt(gamma ** 2 - omega ** 2)
                alpha_p = -gamma + sqrt
                alpha_m = -gamma - sqrt
                scale_1 = np.exp(alpha_p * t) / (alpha_p - alpha_m)
                scale_2 = -np.exp(alpha_m * t) / (alpha_p - alpha_m)

            scale = (lr * (1 - dampening)) * 2
            if np.all(np.isfinite(g_Wl_t_1)) and np.all(np.isfinite(g_bl_t_1)):
                theoretical[layer][step] += (
                    scale
                    * scale_1
                    * utils.in_synapses(g_Wl_t_1, g_bl_t_1, dtype=np.float128)
                )
            if np.all(np.isfinite(g_Wl_t_2)) and np.all(np.isfinite(g_bl_t_2)):
                theoretical[layer][step] += (
                    scale
                    * scale_2
                    * utils.in_synapses(g_Wl_t_2, g_bl_t_2, dtype=np.float128)
                )


def scale(model, feats_dir, steps, **kwargs):
    lr = kwargs.get("lr")
    wd = kwargs.get("wd")

    layers = [layer for layer in utils.get_layers(model) if "conv" in layer]
    W_0 = utils.load_features(
        steps=[str(steps[0])],
        feats_dir=feats_dir,
        model=model,
        suffix="weight",
        group="params",
    )
    b_0 = utils.load_features(
        steps=[str(steps[0])],
        feats_dir=feats_dir,
        model=model,
        suffix="bias",
        group="params",
    )

    load_kwargs = {
        "model": model,
        "feats_dir": feats_dir,
    }
    theory_kwargs = {
        "lr": lr,
        "wd": wd,
        "W_0": W_0,
        "b_0": b_0,
        "step_0": steps[0],
    }

    theoretical = {layer: {} for layer in layers}
    empirical = {layer: {} for layer in layers}
    for i in tqdm(range(len(steps))):
        step = steps[i]
        theory_kwargs["i"] = i
        load_kwargs["group"] = "buffers"
        compute_theoretical(step, layers, load_kwargs, theoretical, **theory_kwargs)
        load_kwargs["group"] = "params"
        compute_empirical(step, layers, load_kwargs, empirical)

    return {"empirical": empirical, "theoretical": theoretical}


def scale_momentum(model, feats_dir, steps, **kwargs):
    lr = kwargs.get("lr")
    wd = kwargs.get("wd")
    momentum = kwargs.get("momentum")
    dampening = kwargs.get("dampening")

    lr = np.array(lr, dtype=np.float128)
    wd = np.array(wd, dtype=np.float128)
    momentum = np.array(momentum, dtype=np.float128)
    dampening = np.array(dampening, dtype=np.float128)

    denom = lr * (1 - dampening) * (1 + momentum)
    gamma = (1 - momentum) / denom
    omega = np.sqrt(4 * wd / denom)

    layers = [layer for layer in utils.get_layers(model) if "conv" in layer]
    W_0 = utils.load_features(
        steps=[str(steps[0])],
        feats_dir=feats_dir,
        model=model,
        suffix="weight",
        group="params",
    )
    b_0 = utils.load_features(
        steps=[str(steps[0])],
        feats_dir=feats_dir,
        model=model,
        suffix="bias",
        group="params",
    )

    load_kwargs = {
        "model": model,
        "feats_dir": feats_dir,
    }
    theory_kwargs = {
        "lr": lr,
        "wd": wd,
        "momentum": momentum,
        "dampening": dampening,
        "gamma": gamma,
        "omega": omega,
        "W_0": W_0,
        "b_0": b_0,
        "step_0": steps[0],
    }

    theoretical = {layer: {} for layer in layers}
    empirical = {layer: {} for layer in layers}
    for i in tqdm(range(len(steps))):
        step = steps[i]
        theory_kwargs["i"] = i
        load_kwargs["group"] = "buffers"
        compute_theoretical_momentum(
            step, layers, load_kwargs, theoretical, **theory_kwargs,
        )
        load_kwargs["group"] = "params"
        compute_empirical(step, layers, load_kwargs, empirical)

    return {"empirical": empirical, "theoretical": theoretical}
