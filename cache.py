import os
import deepdish as dd
import glob
import json
from utils import flags
from metrics import helper
from metrics.metrics import metric_fns


def main(args=None):
    if args is not None:
        ARGS = args

    # load hyperparameters
    with open(
        f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/hyperparameters.json"
    ) as f:
        hyperparameters = json.load(f)
    model = hyperparameters.pop("model")

    # load cache or run metrics
    print(">> Loading weights...")
    cache_path = f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/cache"
    helper.makedir_quiet(cache_path)

    if len(ARGS.metrics) == 0:
        ARGS.metrics = list(metric_fns.keys())
    for metric in ARGS.metrics:
        cache_file = f"{cache_path}/{metric}{ARGS.suffix}.h5"
        if os.path.isfile(cache_file) and not ARGS.overwrite:
            print(f"   Loading {metric} from cache...")
            steps, metrics = dd.io.load(cache_file)
        else:
            step_names = glob.glob(
                f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/feats/*.h5"
            )
            steps = sorted(
                [int(s.split(".h5")[0].split("step")[1]) for s in step_names]
            )
            print(f"   Computing {metric} from extracted features...")
            metrics = metric_fns[metric](
                model=model,
                feats_dir=f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}/feats",
                steps=steps,
                **(hyperparameters),
            )  # TODO: pass subset and seed for network plot
            print(f"   Caching features to {cache_file}")
            dd.io.save(cache_file, (steps, metrics))

    # NOTE: this will only return the last one, for use with plot.py
    return steps, metrics


def validate_cache(parsed_args):
    for m in parsed_args.metrics:
        assert m in list(
            metric_fns.keys()
        ), f"--metrics must be a comma separated list of these options: {','.join(list(metric_fns.keys()))}"


if __name__ == "__main__":
    parser = flags.cache()
    ARGS = parser.parse_args()
    validate_cache(ARGS)

    main(ARGS)
