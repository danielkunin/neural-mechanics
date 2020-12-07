import os
from tqdm import tqdm
import metrics.helper as utils
import numpy as np

from metrics.scale import scale, scale_momentum
from metrics.rescale import rescale, rescale_momentum
from metrics.translation import translation, translation_momentum


def gradient(model, feats_dir, steps, **kwargs):
    layers = [layer for layer in utils.get_layers(model) if "conv" in layer]

    empirical = {layer: {} for layer in layers}
    for i in tqdm(range(len(steps))):
        step = steps[i]
        if step == 0:
            continue
        weight_buffers = utils.load_features(
            steps=[str(step)],
            feats_dir=feats_dir,
            model=model,
            suffix="weight.grad_norm_buffer",
            group="buffers",
        )
        bias_buffers = utils.load_features(
            steps=[str(step)],
            feats_dir=feats_dir,
            model=model,
            suffix="bias.grad_norm_buffer",
            group="buffers",
        )
        for layer in layers:
            wl_t = weight_buffers[layer][f"step_{step}"]
            bl_t = bias_buffers[layer][f"step_{step}"]
            empirical[layer][step] = utils.in_synapses(wl_t, bl_t)

    return {"empirical": empirical}


def network(model, feats_dir, steps, **kwargs):
    subset = kwargs.get("subset", None)
    seed = kwargs.get("seed", 0)
    layers = [layer for layer in utils.get_layers(model)]
    empirical = {layer: {} for layer in layers}
    for i in range(len(steps)):
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
        np.random.seed(seed)
        for layer in layers:
            Wl_t = weights[layer][f"step_{step}"]
            bl_t = biases[layer][f"step_{step}"]
            all_weights = np.concatenate((Wl_t.reshape(-1), bl_t.reshape(-1)))
            if subset is None:
                random_subset_idx = np.arange(len(all_weights))
            else:
                random_subset_idx = np.random.choice(
                    len(all_weights), size=min(subset, len(all_weights)), replace=False
                )
            empirical[layer][step] = all_weights[random_subset_idx]

    return {"empirical": empirical}


def performance(model, feats_dir, steps, **kwargs):
    metrics = {}
    for i in tqdm(range(len(steps))):
        step = steps[i]
        feats_path = f"{feats_dir}/step{step}.h5"
        if os.path.isfile(feats_path):
            feature_dict = utils.get_features(
                feats_path=feats_path,
                group="metrics",
                keys=["accuracy1", "accuracy5", "train_loss", "test_loss"],
            )
            metrics[step] = feature_dict
    return {"performance": metrics}


metric_fns = {
    "scale": scale,
    "rescale": rescale,
    "translation": translation,
    "scale-momentum": scale_momentum,
    "rescale-momentum": rescale_momentum,
    "translation-momentum": translation_momentum,
    "gradient": gradient,
    "performance": performance,
    "network": network,
}
