import os
from tqdm import tqdm
import utils
import numpy as np


def scale(model, feats_dir, steps, **kwargs):
    lr = kwargs.get("lr")
    wd = kwargs.get("wd")

    layers = [layer for layer in utils.get_layers(model) if "conv" in layer]
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

    theoretical = {layer: {} for layer in layers}
    for i in tqdm(range(len(steps))):
        step = steps[i]
        t = lr * step

        if i > 0:
            weight_buffers = utils.load_features(
                steps=[str(step)],
                feats_dir=feats_dir,
                model=model,
                suffix="weight.integral_buffer",
                group="buffers",
            )
            bias_buffers = utils.load_features(
                steps=[str(step)],
                feats_dir=feats_dir,
                model=model,
                suffix="bias.integral_buffer",
                group="buffers",
            )

        for layer in layers:
            Wl_t = weights[layer][f"step_{steps[0]}"]
            bl_t = biases[layer][f"step_{steps[0]}"]
            theoretical[layer][step] = np.exp(-2 * wd * t) * utils.in_synapses(
                Wl_t ** 2, bl_t ** 2
            )
            if i > 0:
                g_Wl_t = weight_buffers[layer][f"step_{step}"]
                g_bl_t = bias_buffers[layer][f"step_{step}"]
                theoretical[layer][step] += (
                    (lr ** 2) * np.exp(-2 * wd * t) * utils.in_synapses(g_Wl_t, g_bl_t)
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
            Wl_t = weights[layer][f"step_{step}"]
            bl_t = biases[layer][f"step_{step}"]
            empirical[layer][step] = utils.in_synapses(Wl_t ** 2, bl_t ** 2)

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

    theoretical = {layer: {} for layer in layers}
    for i in tqdm(range(len(steps))):
        step = steps[i]
        t = lr * (1 - dampening) * step

        if i > 0:
            weight_buffers_1 = utils.load_features(
                steps=[str(step)],
                feats_dir=feats_dir,
                model=model,
                suffix="weight.integral_buffer_1",
                group="buffers",
            )
            bias_buffers_1 = utils.load_features(
                steps=[str(step)],
                feats_dir=feats_dir,
                model=model,
                suffix="bias.integral_buffer_1",
                group="buffers",
            )
            weight_buffers_2 = utils.load_features(
                steps=[str(step)],
                feats_dir=feats_dir,
                model=model,
                suffix="weight.integral_buffer_2",
                group="buffers",
            )
            bias_buffers_2 = utils.load_features(
                steps=[str(step)],
                feats_dir=feats_dir,
                model=model,
                suffix="bias.integral_buffer_2",
                group="buffers",
            )

        for layer in layers:
            Wl_t = weights[layer][f"step_{steps[0]}"]
            bl_t = biases[layer][f"step_{steps[0]}"]
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
                Wl_t ** 2, bl_t ** 2, dtype=np.float128
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
            Wl_t = weights[layer][f"step_{step}"]
            bl_t = biases[layer][f"step_{step}"]
            empirical[layer][step] = utils.in_synapses(Wl_t ** 2, bl_t ** 2)

    return {"empirical": empirical, "theoretical": theoretical}


def rescale(model, feats_dir, steps, **kwargs):
    lr = kwargs.get("lr")
    wd = kwargs.get("wd")

    layers = [layer for layer in utils.get_layers(model) if "conv" in layer]
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

    theoretical = {layer: {} for layer in layers[1:]}
    for i in tqdm(range(len(steps))):
        step = steps[i]
        t = lr * step

        if i > 0:
            weight_buffers = utils.load_features(
                steps=[str(step)],
                feats_dir=feats_dir,
                model=model,
                suffix="weight.integral_buffer",
                group="buffers",
            )
            bias_buffers = utils.load_features(
                steps=[str(step)],
                feats_dir=feats_dir,
                model=model,
                suffix="bias.integral_buffer",
                group="buffers",
            )

        W_in = np.exp(-2 * wd * t) * weights[layers[0]][f"step_{steps[0]}"] ** 2
        b_in = np.exp(-2 * wd * t) * biases[layers[0]][f"step_{steps[0]}"] ** 2
        if i > 0:
            g_W = weight_buffers[layers[0]][f"step_{step}"]
            g_b = bias_buffers[layers[0]][f"step_{step}"]
            W_in += (lr ** 2) * np.exp(-2 * wd * t) * g_W
            b_in += (lr ** 2) * np.exp(-2 * wd * t) * g_b
        for layer in layers[1:]:
            W_out = np.exp(-2 * wd * t) * weights[layer][f"step_{steps[0]}"] ** 2
            b_out = np.exp(-2 * wd * t) * biases[layer][f"step_{steps[0]}"] ** 2
            if i > 0:
                g_W = weight_buffers[layer][f"step_{step}"]
                g_b = bias_buffers[layer][f"step_{step}"]
                W_out += (lr ** 2) * np.exp(-2 * wd * t) * g_W
                b_out += (lr ** 2) * np.exp(-2 * wd * t) * g_b
            theoretical[layer][step] = utils.out_synapses(W_out) - utils.in_synapses(
                W_in, b_in
            )
            W_in = W_out
            b_in = b_out

    empirical = {layer: {} for layer in layers[1:]}
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
        W_in = weights[layers[0]][f"step_{step}"] ** 2
        b_in = biases[layers[0]][f"step_{step}"] ** 2
        for layer in layers[1:]:
            W_out = weights[layer][f"step_{step}"] ** 2
            b_out = biases[layer][f"step_{step}"] ** 2
            empirical[layer][step] = utils.out_synapses(W_out) - utils.in_synapses(
                W_in, b_in
            )
            W_in = W_out
            b_in = b_out

    return {"empirical": empirical, "theoretical": theoretical}


def rescale_momentum(model, feats_dir, steps, **kwargs):
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

    theoretical = {layer: {} for layer in layers[1:]}
    for i in tqdm(range(len(steps))):
        step = steps[i]
        t = lr * (1 - dampening) * step

        if i > 0:
            weight_buffers_1 = utils.load_features(
                steps=[str(step)],
                feats_dir=feats_dir,
                model=model,
                suffix="weight.integral_buffer_1",
                group="buffers",
            )
            bias_buffers_1 = utils.load_features(
                steps=[str(step)],
                feats_dir=feats_dir,
                model=model,
                suffix="bias.integral_buffer_1",
                group="buffers",
            )
            weight_buffers_2 = utils.load_features(
                steps=[str(step)],
                feats_dir=feats_dir,
                model=model,
                suffix="weight.integral_buffer_2",
                group="buffers",
            )
            bias_buffers_2 = utils.load_features(
                steps=[str(step)],
                feats_dir=feats_dir,
                model=model,
                suffix="bias.integral_buffer_2",
                group="buffers",
            )

        W_in = weights[layers[0]][f"step_{steps[0]}"] ** 2
        b_in = biases[layers[0]][f"step_{steps[0]}"] ** 2
        if i > 0:
            g_W_in_1 = weight_buffers_1[layers[0]][f"step_{step}"]
            g_b_in_1 = bias_buffers_1[layers[0]][f"step_{step}"]
            g_W_in_2 = weight_buffers_2[layers[0]][f"step_{step}"]
            g_b_in_2 = bias_buffers_2[layers[0]][f"step_{step}"]

        for layer in layers[1:]:
            W_out = weights[layer][f"step_{steps[0]}"] ** 2
            b_out = biases[layer][f"step_{steps[0]}"] ** 2

            # Homogenous solution
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

            theoretical[layer][step] = scale * (
                utils.out_synapses(W_out, dtype=np.float128)
                - utils.in_synapses(W_in, b_in, dtype=np.float128)
            )
            W_in = W_out
            b_in = b_out

            if i > 0:
                g_W_out_1 = weight_buffers_1[layer][f"step_{step}"]
                g_b_out_1 = bias_buffers_1[layer][f"step_{step}"]
                g_W_out_2 = weight_buffers_2[layer][f"step_{step}"]
                g_b_out_2 = bias_buffers_2[layer][f"step_{step}"]

                # Inhomogenous solution
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

                if (
                    np.all(np.isfinite(g_W_out_1))
                    and np.all(np.isfinite(g_W_in_1))
                    and np.all(np.isfinite(g_b_in_1))
                ):
                    theoretical[layer][step] += (
                        scale
                        * scale_1
                        * (
                            utils.out_synapses(g_W_out_1, dtype=np.float128)
                            - utils.in_synapses(g_W_in_1, g_b_in_1, dtype=np.float128)
                        )
                    )
                if (
                    np.all(np.isfinite(g_W_out_2))
                    and np.all(np.isfinite(g_W_in_2))
                    and np.all(np.isfinite(g_b_in_2))
                ):
                    theoretical[layer][step] += (
                        scale
                        * scale_2
                        * (
                            utils.out_synapses(g_W_out_2, dtype=np.float128)
                            - utils.in_synapses(g_W_in_2, g_b_in_2, dtype=np.float128)
                        )
                    )

                g_W_in_1 = g_W_out_1
                g_b_in_1 = g_b_out_1
                g_W_in_2 = g_W_out_2
                g_b_in_2 = g_b_out_2

    empirical = {layer: {} for layer in layers[1:]}
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
        W_in = weights[layers[0]][f"step_{step}"] ** 2
        b_in = biases[layers[0]][f"step_{step}"] ** 2
        for layer in layers[1:]:
            W_out = weights[layer][f"step_{step}"] ** 2
            b_out = biases[layer][f"step_{step}"] ** 2
            empirical[layer][step] = utils.out_synapses(W_out) - utils.in_synapses(
                W_in, b_in
            )
            W_in = W_out
            b_in = b_out

    return {"empirical": empirical, "theoretical": theoretical}


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


def gradient(model, feats_dir, steps, **kwargs):
    layers = [layer for layer in utils.get_layers(model) if "classifier" in layer]

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
            Wl_t = np.column_stack((wl_t, bl_t))
            empirical[layer][step] = utils.in_synapses(wl_t, bl_t)

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
