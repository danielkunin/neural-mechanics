"""
Creates a plot for projection of columns of weights before softmax at time t and t+k
 over time for fixed k
"""

# mpl imports
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# standard imports
import os
import numpy as np
import deepdish as dd

# convergenets imports
from scripts.visualizations import utils
from scripts.visualizations import helpers
SAVE_BASE = "."
CACHE_DIR = "."


def compute_projection(model, feats_dir, stop, anchor_freq, eta, lamb):
    steps = np.arange(0, stop, anchor_freq)
    features = helpers.load_weights(
        model=model, 
        feats_dir=f"{feats_dir}/feats",
        steps=[str(steps[0])],
        all_steps=False
    )
    layers = list(features.keys())

    theoretical = {layer: {} for layer in layers[0:-1]}
    for i in range(len(steps)):
        step = steps[i]
        t = eta * step
        alpha_p = (-1 + np.sqrt(1 - 4 * eta * lamb)) / eta 
        alpha_m = (-1 - np.sqrt(1 - 4 * eta * lamb)) / eta 
        numer = (alpha_p * np.exp(alpha_m * t) - alpha_m * np.exp(alpha_p * t))
        denom = (alpha_p - alpha_m)

        if i > 0:
            optimizer = helpers.load_optimizer(
                model=model, 
                feats_dir=f"{feats_dir}/feats",
                steps=[str(step)],
                all_steps=False
            )

        for layer in layers[0:-1]:
            Wl_t = features[layer][f"step_{steps[0]}"]
            theoretical[layer][step] = numer / denom * np.sum(Wl_t**2, axis=1)
            if i > 0:
                gl_t_squared = optimizer[layer][f"step_{step}"]
                theoretical[layer][step] += 2 / denom * eta * np.exp(alpha_p * t) * np.sum(gl_t_squared, axis=1)

    empirical = {layer: {} for layer in layers[0:-1]}
    for i in range(len(steps)):
        step = steps[i]
        features = helpers.load_weights(
            model=model, 
            feats_dir=f"{feats_dir}/feats",
            steps=[str(step)],
            all_steps=False
        )
        if f"step_{step}" in features[layers[0]].keys():
            for layer in layers[0:-1]:
                Wl_t = features[layer][f"step_{step}"]
                empirical[layer][step] = np.sum(Wl_t**2, axis=1)
        else:
            print("Feautres don't exist.")

    return (steps, empirical, theoretical)


def main(args=None, axes=None):
    """
    """
    if args is not None:
        ARGS = args

    # load accuracy, loss from db
    print(">> Loading weights...")
    cache_path = f"{CACHE_DIR}/{ARGS.feats_path}/cache"
    utils.makedir_quiet(cache_path)
    cache_file = f"{cache_path}/scale.h5"

    if os.path.isfile(cache_file) and not ARGS.overwrite:
        print("\t\t Loading from cache...")
        steps, empirical, theoretical = dd.io.load(cache_file)
    else:
        steps, empirical, theoretical = compute_projection(
            model=ARGS.model,
            feats_dir=ARGS.feats_path,
            stop=ARGS.stop,
            anchor_freq=ARGS.anchor_freq,
            eta=ARGS.eta,
            lamb=ARGS.lamb,
        )
        print(f"caching features to {cache_file}")
        dd.io.save(cache_file, (steps, empirical, theoretical))

    # plot for each layer
    print(">> Plotting...")
    plt.rcParams["font.size"] = 18
    fig, axes = plt.subplots(figsize=(15, 15))

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
            timesteps,
            norm,
            color=plt.cm.tab10(int(layer.split("fc")[1]) - 1),
        )
    for layer in layers:
        timesteps = list(theoretical[layer].keys())
        norm = list(theoretical[layer].values())
        if args.layer_wise:
            norm = [np.sum(i) for i in norm]
        axes.plot(
            timesteps,
            norm,
            color='k',
            ls='--',
        )
    axes.locator_params(axis="x", nbins=10)
    axes.legend()
    axes.set_xlabel("timestep")
    axes.set_ylabel("squared layer norm")
    axes.title.set_text(
        f"Norm for scale parameters across time"
    )


    if ARGS.use_tex:
        axes.set_xlabel("timestep")
        axes.set_ylabel("squared layer norm")
        axes.set_title(
            r"Norm for scale parameters across time"
        )
    
    # save plot
    plot_path = f"{SAVE_BASE}/{ARGS.feats_path}/img"
    utils.makedir_quiet(plot_path)
    plot_file = f"{plot_path}/scale{ARGS.image_suffix}.pdf"
    plt.savefig(plot_file)
    plt.show()
    print(f">> Saving figure to {plot_file}")


def extend_parser(parser):
    parser.add_argument(
        "--eta",
        type=float,
        help="learning rate",
        required=True,
    )
    parser.add_argument(
        "--lamb", type=float, help="regularization constant", required=True,
    )
    parser.add_argument(
        "--layer-list",
        type=int,
        help="list of layer indices to plot",
        nargs='+',
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
    parser.add_argument(
        "--semilog",
        type=bool,
        help="whether to use a log y-scale",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--ylim",
        type=bool,
        help="whether to set ylim to [0,1]",
        default=False,
        required=False,
    )
    return parser


if __name__ == "__main__":
    parser = helpers.get_default_plot_parser()
    parser = extend_parser(parser)
    ARGS, _ = parser.parse_known_args()

    if ARGS.use_tex:
        from matplotlib import rc

        # For TeX usage in titles
        rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
        ## for Palatino and other serif fonts use:
        # rc('font',**{'family':'serif','serif':['Palatino']})
        rc("text", usetex=True)

    main(ARGS)
