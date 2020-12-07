from tqdm import tqdm
import metrics.helper as utils
import numpy as np


def translation(model, feats_dir, steps, **kwargs):
    lr = kwargs.get("lr")
    wd = kwargs.get("wd")

    layers = [layer for layer in utils.get_layers(model) if "classifier" in layer]
    weights = utils.load_features(
        steps=[str(steps[0])],
        feats_dir=feats_dir,
        model=model,
        suffix="weight",
        group="params",
    )
    biases = utils.load_features(
        steps=[str(steps[0])],
        feats_dir=feats_dir,
        model=model,
        suffix="bias",
        group="params",
    )
    wl_0 = weights["classifier"][f"step_{steps[0]}"]
    bl_0 = biases["classifier"][f"step_{steps[0]}"]
    Wl_0 = np.column_stack((wl_0, bl_0))
    theoretical = {layer: {} for layer in layers}
    for i in tqdm(range(len(steps))):
        step = steps[i]
        t = lr * step
        for layer in layers:
            theoretical[layer][step] = np.exp(-wd * t) * utils.out_synapses(Wl_0)

    empirical = {layer: {} for layer in layers}
    for i in tqdm(range(len(steps))):
        step = steps[i]
        weights = utils.load_features(
            steps=[str(step)],
            feats_dir=feats_dir,
            model=model,
            suffix="weight",
            group="params",
        )
        biases = utils.load_features(
            steps=[str(step)],
            feats_dir=feats_dir,
            model=model,
            suffix="bias",
            group="params",
        )
        for layer in layers:
            wl_t = weights[layer][f"step_{step}"]
            bl_t = biases[layer][f"step_{step}"]
            Wl_t = np.column_stack((wl_t, bl_t))
            empirical[layer][step] = utils.out_synapses(Wl_t)

    return {"empirical": empirical, "theoretical": theoretical}


def translation_momentum(model, feats_dir, steps, **kwargs):
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
    omega = np.sqrt(2 * wd / denom)

    layers = [layer for layer in utils.get_layers(model) if "classifier" in layer]
    weights = utils.load_features(
        steps=[str(steps[0])],
        feats_dir=feats_dir,
        model=model,
        suffix="weight",
        group="params",
    )
    biases = utils.load_features(
        steps=[str(steps[0])],
        feats_dir=feats_dir,
        model=model,
        suffix="bias",
        group="params",
    )
    wl_0 = weights["classifier"][f"step_{steps[0]}"]
    bl_0 = biases["classifier"][f"step_{steps[0]}"]
    Wl_0 = np.column_stack((wl_0, bl_0))
    theoretical = {layer: {} for layer in layers}
    for i in tqdm(range(len(steps))):
        step = steps[i]
        t = lr * (1 - dampening) * step
        for layer in layers:
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

            theoretical[layer][step] = scale * utils.out_synapses(
                Wl_0, dtype=np.float128
            )

    empirical = {layer: {} for layer in layers}
    for i in tqdm(range(len(steps))):
        step = steps[i]
        weights = utils.load_features(
            steps=[str(step)],
            feats_dir=feats_dir,
            model=model,
            suffix="weight",
            group="params",
        )
        biases = utils.load_features(
            steps=[str(step)],
            feats_dir=feats_dir,
            model=model,
            suffix="bias",
            group="params",
        )
        for layer in layers:
            wl_t = weights[layer][f"step_{step}"]
            bl_t = biases[layer][f"step_{step}"]
            Wl_t = np.column_stack((wl_t, bl_t))
            empirical[layer][step] = utils.out_synapses(Wl_t)

    return {"empirical": empirical, "theoretical": theoretical}
