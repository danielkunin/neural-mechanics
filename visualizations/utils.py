import importlib
import os
import sys
import numpy as np
import pprint
import h5py
import glob
import argparse
import math
import socket
import getpass

# This mapping dict needs to be coded manually for every different model we
# want to plot for, and is done via manual, interactive checkpoint inspection.
# This is just meant to identify the layers in the checkpoints
# and map them to pretty, sequential names
PT_MODELS = {
    "logistic": {
        "layers": ["fc1"],
        "weights": ["1.weight"],
        "biases": ["1.bias"],
        "optimizer": ["1.weight"],
    },
    "fc": {
        "layers": [f"fc{x//2 + 1}" for x in range(1, 12, 2)],
        "weights": [f"{x}.weight" for x in range(1, 12, 2)],
        "biases": [f"{x}.bias" for x in range(1, 12, 2)],
        "optimizer": [f"{x}.weight" for x in range(1, 12, 2)],
    },
    "fc_bn": {
        "layers": [f"fc{x//3 + 1}" for x in range(1, 17, 3)],
        "weights": [f"{x}.weight" for x in range(1, 17, 3)],
        "biases": [f"{x}.bias" for x in range(1, 17, 3)],
        "optimizer": [f"{x}.weight" for x in range(1, 17, 3)],
    },
    "res18_bn": {
        "layers": [f"conv{x}" for x in range(1, 19, 1)],
        "weights": [
            "conv1.0.weight",
            "conv2_x.0.residual_function.0.weight",
            "conv2_x.0.residual_function.3.weight",
            "conv2_x.1.residual_function.0.weight",
            "conv2_x.1.residual_function.3.weight",
            "conv3_x.0.residual_function.0.weight",
            "conv3_x.0.residual_function.3.weight",
            "conv3_x.1.residual_function.0.weight",
            "conv3_x.1.residual_function.3.weight",
            "conv4_x.0.residual_function.0.weight",
            "conv4_x.0.residual_function.3.weight",
            "conv4_x.1.residual_function.0.weight",
            "conv4_x.1.residual_function.3.weight",
            "conv5_x.0.residual_function.0.weight",
            "conv5_x.0.residual_function.3.weight",
            "conv5_x.1.residual_function.0.weight",
            "conv5_x.1.residual_function.3.weight",
            "fc.weight",
        ],
        "biases": [
            "conv1.0.bias",
            "conv2_x.0.residual_function.0.bias",
            "conv2_x.0.residual_function.3.bias",
            "conv2_x.1.residual_function.0.bias",
            "conv2_x.1.residual_function.3.bias",
            "conv3_x.0.residual_function.0.bias",
            "conv3_x.0.residual_function.3.bias",
            "conv3_x.1.residual_function.0.bias",
            "conv3_x.1.residual_function.3.bias",
            "conv4_x.0.residual_function.0.bias",
            "conv4_x.0.residual_function.3.bias",
            "conv4_x.1.residual_function.0.bias",
            "conv4_x.1.residual_function.3.bias",
            "conv5_x.0.residual_function.0.bias",
            "conv5_x.0.residual_function.3.bias",
            "conv5_x.1.residual_function.0.bias",
            "conv5_x.1.residual_function.3.bias",
            "fc.bias",
        ],
        "optimizer": [
            "conv1.0.weight",
            "conv2_x.0.residual_function.0.weight",
            "conv2_x.0.residual_function.3.weight",
            "conv2_x.1.residual_function.0.weight",
            "conv2_x.1.residual_function.3.weight",
            "conv3_x.0.residual_function.0.weight",
            "conv3_x.0.residual_function.3.weight",
            "conv3_x.1.residual_function.0.weight",
            "conv3_x.1.residual_function.3.weight",
            "conv4_x.0.residual_function.0.weight",
            "conv4_x.0.residual_function.3.weight",
            "conv4_x.1.residual_function.0.weight",
            "conv4_x.1.residual_function.3.weight",
            "conv5_x.0.residual_function.0.weight",
            "conv5_x.0.residual_function.3.weight",
            "conv5_x.1.residual_function.0.weight",
            "conv5_x.1.residual_function.3.weight",
            "fc.weight",
        ],
    },
}

def makedir_quiet(d):
    """
    Convenience util to create a directory if it doesn't exist
    """
    if not os.path.isdir(d):
        os.makedirs(d)


def make_iterable(x):
    """
    If x is not already array_like, turn it into a list or np.array
    """
    if not isinstance(x, (list, tuple, np.ndarray)):
        return [x]
    return x

def get_layers(model):
    return PT_MODELS[model]["layers"]

def get_features(
    validation_path,
    group_name="imagenet_features",
    keys=("images"),
    out_keys=None,
    verbose=False,
):
    """
    Returns features from HDF5 DataSet

    Inputs
        validation_path (str): where to find the HDF5 dataset
        group_name (str): the group name used for the particular validation
        keys (str or list of strs): which keys to extract from the group.
        out_keys (list of strs): keys for the output dict
    """
    assert os.path.isfile(validation_path), "%s is not a file" % (validation_path)

    keys = make_iterable(keys)

    if out_keys is None:
        out_keys = keys
    out_keys = make_iterable(out_keys)

    assert len(keys) == len(
        out_keys
    ), "Number of keys does not match number of output keys"

    out = {}
    with h5py.File(validation_path, "r") as open_file:
        if verbose:
            keys_to_print = open_file[group_name].keys()
            print("Keys in dataset:")
            pprint.pprint(keys_to_print)

        for in_key, out_key in zip(keys, out_keys):
            out[out_key] = open_file[group_name][in_key][:]
            if verbose:
                print("Extracted %s:" % out_key, out[out_key].shape)

    # if only one input requested, only provide single output
    # if len(out.keys()) == 1:
    #     _, v = out.popitem()
    #     return v
    return out

def load_features(model, feats_dir, group, steps, all_steps=False):
    """
    layers: is the output keys for the layer feats, or computed quantities
    keys: is the actual keys to be read from the h5 file
    feats: is the output dict
    """

    def step_from_path(p):
        step_str = p.split("/")[-1].split(".h5")[0].split("step")[-1]
        return int(step_str)

    if all_steps:
        glob_path = f"{feats_dir}/step*.h5"
        pths = glob.glob(glob_path)
        steps = [step_from_path(path) for path in pths]

    layers = PT_MODELS[model]["layers"]
    keys = PT_MODELS[model][group]

    feats = {layer: {} for layer in layers}

    for step in steps:
        pth = f"{feats_dir}/step{step}.h5"
        
        if os.path.isfile(pth):
            feature_dict = get_features(
                pth, group_name=group, keys=keys, out_keys=layers
            )
            for layer in layers:
                feats[layer][f"step_{step}"] = feature_dict[layer]
    return feats


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=str, required=True, help='name used to save results (default: "")'
    )
    parser.add_argument(
        "--expid", type=str, required=True, help='name used to save results (default: "")'
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help='Directory to save checkpoints and features (default: "Results")',
    )
    parser.add_argument(
        "--plot-dir", 
        type=str, 
        default=None,
        help="Directory to save cache and figures (default: 'results')",
    )
    parser.add_argument(
        "--overwrite", 
        dest="overwrite", 
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--image-suffix",
        type=str,
        default="",
        help="extra image suffix",
        required=False,
    )
    parser.add_argument(
        "--use-tex", 
        action="store_true",
        help="will use tex rendering for matplotlib labels", 
        default=False
    )
    parser.add_argument(
        "--legend", 
        action="store_true",
        help="will add legend", 
        default=False
    )
    return parser