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

    empirical = {layer: {} for layer in layers}
    for step in steps:
        if step == 0:
            continue
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
        for layer in layers:
            wl_t = weight_buffers[layer][f"step_{step}"]
            bl_t = bias_buffers[layer][f"step_{step}"]
            Wl_t = np.column_stack((wl_t, bl_t))
            empirical[layer][step] = utils.in_synapses(wl_t, bl_t)

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
    cache_file = f"{cache_path}/gradient{ARGS.image_suffix}.h5"
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
    for layer in layers:
        timesteps = list(empirical[layer].keys())
        norm = list(empirical[layer].values())
        if args.layer_wise:
            norm = [np.sum(i) for i in norm]
        axes.plot(
            timesteps, norm, color=plt.cm.tab20(int(layer.split("conv")[1]) - 1),
        )

    # axes labels and title
    axes.set_xlabel("timestep")
    axes.set_ylabel(f"gradient norm")
    axes.title.set_text(f"Gradient norms across time")
    if ARGS.use_tex:
        axes.set_xlabel("timestep")
        axes.set_ylabel(r"$\|g_t\|^2$")
        axes.set_title(r"Gradient norms across time")

    if ARGS.legend:
        axes.legend()

    # save plot
    if ARGS.plot_dir is None:
        plot_path = f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/img"
    else:
        plot_path = f"{ARGS.plot_dir}/img"
    utils.makedir_quiet(plot_path)
    plot_file = f"{plot_path}/gradient{ARGS.image_suffix}.pdf"
    plt.savefig(plot_file)
    print(f">> Saving figure to {plot_file}")


# plot-specific args
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
