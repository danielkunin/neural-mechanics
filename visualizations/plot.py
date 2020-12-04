import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import utils
from cache import main as cache
from cache import metric_fns


y_labels = {
    "scale": "projection",
    "rescale": "projection",
    "translation": "projection",
    "scale-momentum": "projection",
    "rescale-momentum": "projection",
    "translation-momentum": "projection",
    "gradient": "gradient norm squared",
}

y_labels_tex = {
    "scale": r"$\| W_l \|_2^2$",
    "rescale": r"$\| W_l \|_2^2-\| W_{l-1} \|_2^2$",
    "translation": r"$\langle W, 1\rangle$",
    "scale-momentum": r"$\| W_l \|_2^2$",
    "rescale-momentum": r"$\| W_l \|_2^2-\| W_{l-1} \|_2^2$",
    "translation-momentum": r"$\langle W, 1\rangle$",
    "gradient": r"$\| \nabla W\|_2^2$",
}

titles = {
    "scale": "Projection for scale parameters across time (SGD)",
    "rescale": "Projection for rescale parameters across time (SGD)",
    "translation": "Projection for translational parameters across time (SGD)",
    "scale-momentum": "Projection for scale parameters across time (SGDM)",
    "rescale-momentum": "Projection for rescale parameters across time (SGDM)",
    "translation-momentum": "Projection for translational parameters across time (SGDM)",
    "gradient": "Gradient norms across time",
}


def empirical_plot(args, axes, metrics):
    # plot data
    empirical = metrics["empirical"]
    if args.layer_list == None:
        layers = list(empirical.keys())
    else:
        layers = [list(empirical.keys())[i] for i in args.layer_list]
    for layer in layers:
        if "translation" in args.viz:
            color_idx = 1
        else:
            color_idx = int(layer.split("conv")[1]) - 1
        timesteps = list(empirical[layer].keys())
        norm = list(empirical[layer].values())
        if args.layer_wise:
            norm = [np.sum(i) for i in norm]
        axes.plot(
            timesteps, norm, color=plt.cm.tab20(color_idx),
        )

    if "theoretical" in metrics.keys():
        theoretical = metrics["theoretical"]
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
    axes.set_ylabel(y_labels[ARGS.viz])
    axes.title.set_text(titles[ARGS.viz])
    if args.use_tex:
        axes.set_ylabel(y_labels_tex[ARGS.viz])


def performance_plot(axes, steps, performance):
    color = "k"
    plot_steps = []
    for step in steps:
        if "train_loss" in performance[step].keys():
            plot_steps.append(step)
    axes.plot(
        plot_steps, [performance[s]["train_loss"] for s in plot_steps], color=color,
    )
    axes.plot(
        plot_steps,
        [performance[s]["test_loss"] for s in plot_steps],
        color=color,
        alpha=0.5,
    )
    axes.tick_params(axis="y", labelcolor=color)
    axes.set_ylabel(f"loss")

    axes2 = axes.twinx()
    color = "tab:blue"
    axes2.plot(
        plot_steps, [performance[s]["accuracy1"] for s in plot_steps], color=color,
    )
    axes2.plot(
        plot_steps,
        [performance[s]["accuracy5"] for s in plot_steps],
        color=color,
        alpha=0.5,
    )
    axes2.tick_params(axis="y", labelcolor=color)
    axes2.set_ylabel(f"accuracy")

    # axes labels and title
    axes.set_xlabel("train step")
    axes.title.set_text(f"Performance for model over training time")


def network_plot(args, axes, empirical):
    if args.layer_list == None:
        layers = list(empirical.keys())
    else:
        layers = [list(empirical.keys())[i] for i in args.layer_list]

    handles = []
    layers = [l for l in layers if "conv" in l]
    for idx, layer in enumerate(layers):
        timesteps = list(empirical[layer].keys())
        norm = list(empirical[layer].values())
        if args.norm:
            norm = [i ** 2 for i in norm]
        if args.layer_wise:
            norm = [np.sum(i) for i in norm]
        axes.plot(
            timesteps, norm, color=plt.cm.tab20(idx), label=layer, lw=2, alpha=0.5,
        )
        handles += [mpatches.Patch(color=plt.cm.tab20(idx), label=layer)]


def main(args=None, axes=None):
    if args is not None:
        ARGS = args
    steps, metrics = cache(ARGS)

    # create plot
    print(">> Plotting...")
    plt.rcParams["font.size"] = 18
    if axes is None:
        fig, axes = plt.subplots(figsize=(15, 15))

    if "performance" in metrics.keys():
        performance_plot(axes, steps, metrics["performance"])
    elif "network" in ARGS.viz:
        network_plot(ARGS, axes, metrics["empirical"])
    else:
        empirical_plot(ARGS, axes, metrics)

    if ARGS.legend:
        axes.legend()

    # save plot
    if ARGS.plot_dir is None:
        plot_path = f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/img"
    else:
        plot_path = f"{ARGS.plot_dir}/img"
    utils.makedir_quiet(plot_path)
    plot_file = f"{plot_path}/{ARGS.viz}{ARGS.suffix}.pdf"
    plt.savefig(plot_file)
    print(f">> Saving figure to {plot_file}")


def extend_parser(parser):
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save figures (default: <save-dir>/<experiment>/<expid>/img )",
    )
    parser.add_argument(
        "--use-tex",
        action="store_true",
        help="will use tex rendering for matplotlib labels",
        default=False,
    )
    parser.add_argument(
        "--legend", action="store_true", help="will add legend", default=False
    )

    # viz specific: maybe in subparser?
    # NOTE: the normalize arg was not being used anywhere...
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
    parser.add_argument(
        "viz", type=str, choices=list(metric_fns.keys()), help="visualization to plot",
    )
    # subparsers here?? Probably not, don't really need diferent options for each viz

    ARGS = parser.parse_args()

    if ARGS.use_tex:
        from matplotlib import rc

        # For TeX usage in titles
        rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
        ## for Palatino and other serif fonts use:
        # rc('font',**{'family':'serif','serif':['Palatino']})
        rc("text", usetex=True)

    main(ARGS)
