import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
import deepdish as dd
import utils
import glob
import json
from cache import main as cache
from cache import metric_fns


def main(args=None, axes=None):
    steps, metrics = cache(args)

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

    # viz specific: maybe in subparser
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
    # subparsers here??

    ARGS = parser.parse_args()

    if ARGS.use_tex:
        from matplotlib import rc

        # For TeX usage in titles
        rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
        ## for Palatino and other serif fonts use:
        # rc('font',**{'family':'serif','serif':['Palatino']})
        rc("text", usetex=True)

    main(ARGS)
