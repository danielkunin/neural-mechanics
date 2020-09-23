import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
import deepdish as dd
import utils
import glob
import json


def statistics(model, feats_dir, steps, lr, wd):
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
    for i in range(len(steps)):
        step = steps[i]
        t = lr * step
        alpha_p = (-1 + np.sqrt(1 - 4 * lr * wd)) / lr
        alpha_m = (-1 - np.sqrt(1 - 4 * lr * wd)) / lr
        numer = alpha_p * np.exp(alpha_m * t) - alpha_m * np.exp(alpha_p * t)
        denom = alpha_p - alpha_m

        if i > 0:
            weight_buffers = utils.load_features(
                steps=[str(step)],
                feats_dir=feats_dir,
                model=model,
                suffix="weight",
                group="buffers",
            )
            bias_buffers = utils.load_features(
                steps=[str(step)],
                feats_dir=feats_dir,
                model=model,
                suffix="bias",
                group="buffers",
            )

        W_in = numer / denom * weights[layers[0]][f"step_{steps[0]}"] ** 2
        b_in = numer / denom * biases[layers[0]][f"step_{steps[0]}"] ** 2
        if i > 0:
            g_W = weight_buffers[layers[0]][f"step_{step}"]
            g_b = bias_buffers[layers[0]][f"step_{step}"]
            W_in += 2 / denom * lr * np.exp(alpha_p * t) * g_W
            b_in += 2 / denom * lr * np.exp(alpha_p * t) * g_b
        for layer in layers[1:]:
            W_out = numer / denom * weights[layer][f"step_{steps[0]}"] ** 2
            b_out = numer / denom * biases[layer][f"step_{steps[0]}"] ** 2
            if i > 0:
                g_W = weight_buffers[layer][f"step_{step}"]
                g_b = bias_buffers[layer][f"step_{step}"]
                W_out += 2 / denom * lr * np.exp(alpha_p * t) * g_W
                b_out += 2 / denom * lr * np.exp(alpha_p * t) * g_b
            theoretical[layer][step] = utils.out_synapses(W_out) - utils.in_synapses(
                W_in, b_in
            )
            W_in = W_out
            b_in = b_out

    empirical = {layer: {} for layer in layers[1:]}
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

    return (empirical, theoretical)


def main(args=None, axes=None):

    if args is not None:
        ARGS = args
    if ARGS.plot_dir is None:
        ARGS.plot_dir = ARGS.save_dir

    # load hyperparameters
    with open(
        f"{ARGS.plot_dir}/{ARGS.experiment}/{ARGS.expid}/hyperparameters.json"
    ) as f:
        hyperparameters = json.load(f)

    # load cache or run statistics
    print(">> Loading weights...")
    cache_path = f"{ARGS.plot_dir}/{ARGS.experiment}/{ARGS.expid}/cache"
    utils.makedir_quiet(cache_path)
    cache_file = f"{cache_path}/inversion.h5"
    if os.path.isfile(cache_file) and not ARGS.overwrite:
        print("   Loading from cache...")
        steps, empirical, theoretical = dd.io.load(cache_file)
    else:
        step_names = glob.glob(
            f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/feats/*.h5"
        )
        steps = sorted([int(s.split(".h5")[0].split("step")[1]) for s in step_names])
        empirical, theoretical = statistics(
            model=hyperparameters["model"],
            feats_dir=f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/feats",
            steps=steps,
            lr=hyperparameters["lr"],
            wd=hyperparameters["wd"],
        )
        print(f"   Caching features to {cache_file}")
        dd.io.save(cache_file, (steps, empirical, theoretical))

    # create plot
    print(">> Plotting...")
    plt.rcParams["font.size"] = 18
    if axes is None:
        fig, axes = plt.subplots(figsize=(15, 15))

    # plot data
    if args.layer_list == None:
        layers = list(empirical.keys())
    else:
        layers = [list(empirical.keys())[i] for i in args.layer_list]
    for layer in layers:
        timesteps = list(empirical[layer].keys())
        norm = list(empirical[layer].values())
        if args.layer_wise:
            norm = [np.sum(i) for i in norm]
        axes.plot(
            timesteps, norm, color=plt.cm.tab20(int(layer.split("conv")[1]) - 1),
        )
    for layer in layers:
        timesteps = list(theoretical[layer].keys())
        norm = list(theoretical[layer].values())
        if args.layer_wise:
            norm = [np.sum(i) for i in norm]
        axes.plot(
            timesteps, norm, color="k", ls="--",
        )

    # axes labels and title
    axes.set_xlabel("timestep")
    axes.set_ylabel(f"projection")
    axes.title.set_text(f"Projection for translational parameters across time")
    if ARGS.use_tex:
        axes.set_xlabel("timestep")
        axes.set_ylabel(r"$\langle W, \mathbb{1}\rangle$")
        axes.set_title(r"Projection for translational parameters across time")

    if ARGS.legend:
        axes.legend()

    # save plot
    plot_path = f"{ARGS.plot_dir}/{ARGS.experiment}/{ARGS.expid}/img"
    utils.makedir_quiet(plot_path)
    plot_file = f"{plot_path}/inversion{ARGS.image_suffix}.pdf"
    plt.savefig(plot_file)
    print(f">> Saving figure to {plot_file}")


def extend_parser(parser):
    parser.add_argument(
        "--layer-list",
        type=int,
        help="list of layer indices to plot",
        nargs="+",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--layer-wise",
        type=bool,
        help="whether to plot per neuron",
        default=False,
        required=False,
    )
    return parser


if __name__ == "__main__":
    parser = utils.default_parser()
    parser = extend_parser(parser)
    ARGS = parser.parse_args()

    if ARGS.use_tex:
        from matplotlib import rc

        # For TeX usage in titles
        rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
        ## for Palatino and other serif fonts use:
        # rc('font',**{'family':'serif','serif':['Palatino']})
        rc("text", usetex=True)

    main(ARGS)
