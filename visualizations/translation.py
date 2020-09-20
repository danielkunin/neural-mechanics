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
import utils
SAVE_BASE = "."
CACHE_DIR = "."


def compute_projection(model, feats_dir, stop, anchor_freq, eta, lamb, normalize=False):
    steps = np.arange(0, stop + anchor_freq, anchor_freq)
    features = utils.load_features(
        model=model, 
        feats_dir=f"{feats_dir}/feats",
        group="weights",
        steps=[str(steps[0])],
        all_steps=False
    )
    layers = list(features.keys())
    Wl_0 = features[layers[-1]][f"step_{steps[0]}"]

    theoretical = []
    for i in range(len(steps)):
        t = 1.0 * eta * steps[i]
        alpha_p = (-1 + np.sqrt(1 - 2 * eta * lamb)) / eta 
        alpha_m = (-1 - np.sqrt(1 - 2 * eta * lamb)) / eta 
        numer = (alpha_p * np.exp(alpha_m * t) - alpha_m * np.exp(alpha_p * t))
        denom = (alpha_p - alpha_m)
        if normalize:
            theoretical.append(numer / denom)
            #theoretical.append(np.exp(-lamb * t))
        else:
            theoretical.append(numer / denom * np.sum(Wl_0, axis=0))
            # theoretical.append(np.exp(-lamb * t) * np.sum(Wl_0, axis=0))

    empirical = []
    for i in range(len(steps)):
        step = steps[i]
        features = utils.load_features(
            model=model, 
            feats_dir=f"{feats_dir}/feats",
            group="weights",
            steps=[str(step)],
            all_steps=False
        )
        if f"step_{step}" in features[layers[0]].keys():
            Wl_t = features[layers[-1]][f"step_{step}"]
            if normalize:
                empirical.append(np.sum(Wl_t, axis=0) / np.sum(Wl_0, axis=0))
            else:
                empirical.append(np.sum(Wl_t, axis=0))
        else:
            print(f"Feautres for step_{step} don't exist.")

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
    cache_file = f"{cache_path}/translation.h5"

    if os.path.isfile(cache_file) and not ARGS.overwrite:
        print("\t Loading from cache...")
        steps, empirical, theoretical = dd.io.load(cache_file)
    else:
        steps, empirical, theoretical = compute_projection(
            model=ARGS.model,
            feats_dir=ARGS.feats_path,
            stop=ARGS.stop,
            anchor_freq=ARGS.anchor_freq,
            eta=ARGS.eta,
            lamb=ARGS.lamb,
            normalize=ARGS.normalize,
        )
        print(f"\t Caching features to {cache_file}")
        dd.io.save(cache_file, (steps, empirical, theoretical))

    # plot for each layer
    print(">> Plotting...")
    plt.rcParams["font.size"] = 18
    if ARGS.normalize:
        fig, axes = plt.subplots(figsize=(15, 15))
        axes.plot(steps[0 : len(empirical)], empirical, c="r", ls="-", alpha=0.1)
        axes.plot(
            steps[0 : len(theoretical)],
            theoretical,
            c="b",
            lw=3,
            ls="--",
            label="theoretical",
        )

        axes.locator_params(axis="x", nbins=10)
        axes.legend()
        axes.set_xlabel("timestep")
        axes.set_ylabel(f"projection")
        if ARGS.ylim:
            axes.set_ylim(0.0,0.01)
        if ARGS.semilog:
            axes.set_yscale("log")
        axes.title.set_text(
            f"Projection for translational parameters across time"
        )
    else:
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(30, 15))
        axes[0].plot(steps[0 : len(empirical)], empirical, c="r", ls="-", alpha=0.5)
        axes[1].plot(
            steps[0 : len(theoretical)], theoretical, c="b", ls="--", alpha=0.5
        )

        axes[0].title.set_text("Empirical")
        axes[1].title.set_text("Theoretical")
        for ax in axes:
            ax.locator_params(axis="x", nbins=10)
            ax.set_xlabel("timestep")
            ax.set_ylabel(f"projection")
            if ARGS.ylim:
                ax.set_ylim(-1.1,1.1)
            if ARGS.semilog:
                ax.set_yscale("log")

    if ARGS.use_tex:
        axes.set_xlabel("timestep")
        axes.set_ylabel(r"$\langle W\mathbb{1}\rangle$")
        axes.set_title(
            r"Projection for translational parameters across time"
        )

    # save plot
    plot_path = f"{SAVE_BASE}/{ARGS.feats_path}/img"
    utils.makedir_quiet(plot_path)
    plot_file = f"{plot_path}/translation{ARGS.image_suffix}.pdf"
    plt.savefig(plot_file)
    plt.show()
    print(f">> Saving figure to {plot_file}")


# plot-specific args
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
        "--normalize",
        type=bool,
        help="whether to normalize by initial condition",
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
    parser = utils.get_default_plot_parser()
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
