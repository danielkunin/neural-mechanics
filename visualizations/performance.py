import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
import deepdish as dd
import utils
import glob
import json


def statistics(model, feats_dir, steps):
    statistics = {}
    for i in range(len(steps)):
        step = steps[i]
        feats_path = f"{feats_dir}/step{step}.h5"
        if os.path.isfile(feats_path):
            feature_dict = utils.get_features(
                feats_path=feats_path,
                group="metrics",
                keys=["accuracy1", "accuracy5", "train_loss", "test_loss"],
            )
            statistics[step] = feature_dict
    return statistics


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
    cache_file = f"{cache_path}/performance{ARGS.image_suffix}.h5"
    if os.path.isfile(cache_file) and not ARGS.overwrite:
        print("   Loading from cache...")
        performance = dd.io.load(cache_file)
    else:
        step_names = glob.glob(
            f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/feats/*.h5"
        )
        steps = sorted([int(s.split(".h5")[0].split("step")[1]) for s in step_names])
        performance = statistics(
            model=hyperparameters["model"],
            feats_dir=f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/feats",
            steps=steps,
        )
        print(f"   Caching features to {cache_file}")
        dd.io.save(cache_file, (steps, performance))

    # create plot
    print(">> Plotting...")
    plt.rcParams["font.size"] = 18
    if axes is None:
        fig, axes = plt.subplots(figsize=(15, 15))

    # plot data
    color = "k"
    axes.plot(
        steps, performance["train_loss"], color=color,
    )
    axes.plot(
        steps, performance["test_loss"], color=color, alpha=0.5,
    )
    axes2.tick_params(axis="y", labelcolor=color)
    axes.set_ylabel(f"loss")

    axes2 = axes.twinx()
    color = "tab:blue"
    axes2.plot(
        steps, performance["accuracy1"], color=color,
    )
    axes2.plot(
        steps, performance["accuracy5"], color=color, alpha=0.5,
    )
    axes2.tick_params(axis="y", labelcolor=color)
    axes2.set_ylabel(f"accuracy")

    # axes labels and title
    axes.set_xlabel("train step")
    axes.title.set_text(f"Performance for model over training time")

    if ARGS.legend:
        axes.legend()

    # save plot
    if ARGS.plot_dir is None:
        plot_path = f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/img"
    else:
        plot_path = f"{ARGS.plot_dir}/img"
    utils.makedir_quiet(plot_path)
    plot_file = f"{plot_path}/performance{ARGS.image_suffix}.pdf"
    plt.savefig(plot_file)
    print(f">> Saving figure to {plot_file}")


# plot-specific args
def extend_parser(parser):
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
