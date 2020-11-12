import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np
import deepdish as dd
import utils
import glob
import json


def statistics(model, feats_dir, steps, lr, wd):
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
        for layer in layers:
            Wl_t = weights[layer][f"step_{step}"]
            bl_t = biases[layer][f"step_{step}"]
            empirical[layer][step] = np.concatenate((Wl_t.reshape(-1), bl_t.reshape(-1)))

    return empirical


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
    cache_file = f"{cache_path}/network{ARGS.image_suffix}.h5"
    if os.path.isfile(cache_file) and not ARGS.overwrite:
        print("   Loading from cache...")
        steps, empirical = dd.io.load(cache_file)
    else:
        step_names = glob.glob(
            f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/feats/*.h5"
        )
        steps = sorted([int(s.split(".h5")[0].split("step")[1]) for s in step_names])
        empirical = statistics(
            model=hyperparameters["model"],
            feats_dir=f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/feats",
            steps=steps,
            lr=hyperparameters["lr"],
            wd=hyperparameters["wd"],
        )
        print(f"   Caching features to {cache_file}")
        dd.io.save(cache_file, (steps, empirical))

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

    handles = []
    for idx, layer in enumerate(layers):
        timesteps = list(empirical[layer].keys())
        norm = list(empirical[layer].values())
        if args.norm:
            norm = [i**2 for i in norm]
        if args.subset > 0:
            norm = [i[0:args.subset] for i in norm]
        axes.plot(
            timesteps, norm, color=plt.cm.tab20(idx),
        )
        handles += [mpatches.Patch(color=plt.cm.tab20(idx), label=layer)]

    # axes labels and title
    axes.set_xlabel("timestep")
    axes.set_ylabel(f"projection")
    axes.title.set_text(f"Projection for translational parameters across time")
    if ARGS.use_tex:
        axes.set_xlabel("timestep")
        axes.set_ylabel(r"$\langle W, \mathbb{1}\rangle$")
        axes.set_title(r"Projection for translational parameters across time")
    axes.legend(handles=handles)

    # save plot
    if ARGS.plot_dir is None:
        plot_path = f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/img"
    else:
        plot_path = f"{ARGS.plot_dir}/img"
    utils.makedir_quiet(plot_path)
    plot_file = f"{plot_path}/network{ARGS.image_suffix}.pdf"
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
        "--norm",
        type=bool,
        help="whether to plot squared norm",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--subset",
        type=int,
        help="number of parameters to plot",
        default=None,
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
