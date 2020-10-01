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
    for i in range(len(steps)):
        step = steps[i]
        t = lr * step
        for layer in layers:
            theoretical[layer][step] = np.exp(-wd * t) * utils.out_synapses(Wl_0)

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
        for layer in layers:
            wl_t = weights[layer][f"step_{step}"]
            bl_t = biases[layer][f"step_{step}"]
            Wl_t = np.column_stack((wl_t, bl_t))
            empirical[layer][step] = utils.out_synapses(Wl_t)

    return (empirical, theoretical)


def main(args=None, axes=None):

    if args is not None:
        ARGS = args

    # load hyperparameters
    with open(
        f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/hyperparameters.json"
    ) as f:
        hyperparameters = json.load(f)

    # load cache or run statistics
    print(">> Loading weights...")
    cache_path = f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/cache"
    utils.makedir_quiet(cache_path)
    cache_file = f"{cache_path}/translation.h5"
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
    layers = list(empirical.keys())
    for layer in layers:
        timesteps = list(empirical[layer].keys())
        norm = list(empirical[layer].values())
        if ARGS.layer_wise:
            norm = [np.sum(i) for i in norm]
        axes.plot(
            timesteps, norm,
        )
    for layer in layers:
        timesteps = list(theoretical[layer].keys())
        norm = list(theoretical[layer].values())
        if ARGS.layer_wise:
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
    if ARGS.plot_dir is None:
        plot_path = f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/img"
    else:
        plot_path = f"{ARGS.plot_dir}/img"
    utils.makedir_quiet(plot_path)
    plot_file = f"{plot_path}/translation{ARGS.image_suffix}.pdf"
    plt.savefig(plot_file)
    print(f">> Saving figure to {plot_file}")


# plot-specific args
def extend_parser(parser):
    parser.add_argument(
        "--normalize",
        type=bool,
        help="whether to normalize by initial condition",
        default=False,
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
