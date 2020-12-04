import os
import deepdish as dd
import utils
import glob
import json
from metrics import *

metric_fns = {
    "scale": scale,
    "rescale": rescale,
    "translation": translation,
    "scale-momentum": scale_momentum,
    "rescale-momentum": rescale_momentum,
    "translation-momentum": translation_momentum,
    "gradient": gradient,
    "performance": performance,
}


def main(args=None):
    if args is not None:
        ARGS = args

    # load hyperparameters
    with open(
        f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/hyperparameters.json"
    ) as f:
        hyperparameters = json.load(f)

    # load cache or run metrics
    print(">> Loading weights...")
    cache_path = f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/cache"
    utils.makedir_quiet(cache_path)
    cache_file = f"{cache_path}/{ARGS.viz}{ARGS.suffix}.h5"
    if os.path.isfile(cache_file) and not ARGS.overwrite:
        print("   Loading from cache...")
        steps, metrics = dd.io.load(cache_file)
    else:
        step_names = glob.glob(
            f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/feats/*.h5"
        )
        steps = sorted([int(s.split(".h5")[0].split("step")[1]) for s in step_names])
        print("   Computing metrics from extracted features...")
        model = hyperparameters.pop("model")
        metrics = metric_fns[ARGS.viz](
            model=model,
            feats_dir=f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/feats",
            steps=steps,
            **(hyperparameters),
        )
        print(f"   Caching features to {cache_file}")
        dd.io.save(cache_file, (steps, metrics))

    return steps, metrics


if __name__ == "__main__":
    parser = utils.default_parser()
    parser.add_argument(
        "viz",
        type=str,
        choices=list(metric_fns.keys()),
        help="visualization to generate a cache for",
    )
    ARGS = parser.parse_args()

    main(ARGS)
