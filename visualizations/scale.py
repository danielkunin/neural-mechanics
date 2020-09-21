import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
import deepdish as dd
import utils
import glob


def statistics(model, feats_dir, steps, lr, wd):
    layers = utils.get_layers(model)
    weights = utils.load_features(
        model=model, 
        feats_dir=feats_dir,
        group="weights",
        steps=[str(steps[0])]
    )
    biases = utils.load_features(
        model=model, 
        feats_dir=feats_dir,
        group="biases",
        steps=[str(steps[0])]
    )

    theoretical = {layer: {} for layer in layers[0:-1]}
    for i in range(len(steps)):
        step = steps[i]
        t = lr * step
        alpha_p = (-1 + np.sqrt(1 - 4 * lr * wd)) / lr 
        alpha_m = (-1 - np.sqrt(1 - 4 * lr * wd)) / lr 
        numer = (alpha_p * np.exp(alpha_m * t) - alpha_m * np.exp(alpha_p * t))
        denom = (alpha_p - alpha_m)

        if i > 0:
            optimizer = utils.load_features(
                model=model, 
                feats_dir=feats_dir,
                group="optimizer",
                steps=[str(step)]
            )

        for layer in layers[0:-1]:
            Wl_t = weights[layer][f"step_{steps[0]}"]
            theoretical[layer][step] = numer / denom * np.sum(Wl_t**2, axis=1)
            if i > 0:
                gl_t_squared = optimizer[layer][f"step_{step}"]
                theoretical[layer][step] += 2 / denom * lr * np.exp(alpha_p * t) * np.sum(gl_t_squared, axis=1)

    empirical = {layer: {} for layer in layers[0:-1]}
    for i in range(len(steps)):
        step = steps[i]
        weights = utils.load_features(
            model=model, 
            feats_dir=feats_dir,
            group="weights",
            steps=[str(step)]
        )
        if f"step_{step}" in weights[layers[0]].keys():
            for layer in layers[0:-1]:
                Wl_t = weights[layer][f"step_{step}"]
                empirical[layer][step] = np.sum(Wl_t**2, axis=1)
        else:
            print("Feautres don't exist.")

    return (empirical, theoretical)


def main(args=None, axes=None):

    if args is not None:
        ARGS = args
    if ARGS.plot_dir is None:
        ARGS.plot_dir = ARGS.save_dir

    # load hyperparameters
    with open(f"{ARGS.plot_dir}/{ARGS.experiment}/{ARGS.expid}/args.json") as f:
        hyperparameters = json.load(f)

    # load cache or run statistics
    print(">> Loading weights...")
    cache_path = f"{ARGS.plot_dir}/{ARGS.experiment}/{ARGS.expid}/cache"
    utils.makedir_quiet(cache_path)
    cache_file = f"{cache_path}/scale.h5"
    if os.path.isfile(cache_file) and not ARGS.overwrite:
        print("   Loading from cache...")
        steps, empirical, theoretical = dd.io.load(cache_file)
    else:
        step_names = glob.glob(f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/feats/*.h5")
        steps = sorted([int(s.split(".h5")[0].split("step")[1]) for s in step_names])
        empirical, theoretical = statistics(
            model=hyperparameters['model'],
            feats_dir=f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/feats",
            steps=steps,
            lr=hyperparameters['lr'],
            wd=hyperparameters['weight_decay'],
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
    
    # axes labels and title
    axes.set_xlabel("timestep")
    axes.set_ylabel(f"projection")
    axes.title.set_text(
        f"Projection for translational parameters across time"
    )
    if ARGS.use_tex:
        axes.set_xlabel("timestep")
        axes.set_ylabel(r"$\langle W, \mathbb{1}\rangle$")
        axes.set_title(
            r"Projection for translational parameters across time"
        )

    if ARGS.legend:
        axes.legend()

    # save plot
    plot_path = f"{ARGS.plot_dir}/{ARGS.experiment}/{ARGS.expid}/img"
    utils.makedir_quiet(plot_path)
    plot_file = f"{plot_path}/scale{ARGS.image_suffix}.pdf"
    plt.savefig(plot_file)
    print(f">> Saving figure to {plot_file}")


def extend_parser(parser):
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
