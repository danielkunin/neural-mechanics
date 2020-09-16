"""
Helper utilities for plotting scripts
"""

import os
import glob
import argparse
import math

from scripts.visualizations import file_utils  

# This mapping dict needs to be coded manually for every different model we
# want to plot for, and is done via manual, interactive checkpoint inspection.
# This is just meant to identify the layers in the checkpoints
# and map them to pretty, sequential names
PT_MODELS = {
    "logistic": {
        "layers": ["fc1"],
        "keys": ["1.weight"],
    },
    "fc": {
        "layers": [f"fc{x//2 + 1}" for x in range(1, 12, 2)],
        "keys": [f"{x}.weight" for x in range(1, 12, 2)],
    },
    "fc_bn": {
        "layers": [f"fc{x//3 + 1}" for x in range(1, 17, 3)],
        "keys": [f"{x}.weight" for x in range(1, 17, 3)],
    },
    "res18_bn": {
        "layers": [f"conv{x}" for x in range(1, 19, 1)],
        "keys": [
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


def load_weights(model, feats_dir, steps, all_steps=False):
    """
    layers: is the output keys for the layer weights, or computed quantities
    keys: is the actual keys to be read from the h5 file
    weights: is the output dict
    """

    def step_from_path(p):
        step_str = p.split("/")[-1].split(".h5")[0].split("step")[-1]
        return int(step_str)

    if all_steps:
        glob_path = f"{feats_dir}/step*.h5"
        pths = glob.glob(glob_path)
        steps = [step_from_path(path) for path in pths]

    layers = PT_MODELS[model]["layers"]
    keys = PT_MODELS[model]["keys"]

    weights = {layer: {} for layer in layers}

    for step in steps:
        pth = f"{feats_dir}/step{step}.h5"
        
        if os.path.isfile(pth):
            feature_dict = file_utils.get_features(
                pth, group_name="weights", keys=keys, out_keys=layers
            )
            for layer in layers:
                weights[layer][f"step_{step}"] = feature_dict[layer]
    return weights


def load_optimizer(model, feats_dir, steps, all_steps=False):
    """
    layers: is the output keys for the layer weights, or computed quantities
    keys: is the actual keys to be read from the h5 file
    weights: is the output dict
    """

    def step_from_path(p):
        step_str = p.split("/")[-1].split(".h5")[0].split("step")[-1]
        return int(step_str)

    if all_steps:
        glob_path = f"{feats_dir}/step*.h5"
        pths = glob.glob(glob_path)
        steps = [step_from_path(path) for path in pths]

    layers = PT_MODELS[model]["layers"]
    keys = PT_MODELS[model]["keys"]

    weights = {layer: {} for layer in layers}

    for step in steps:
        pth = f"{feats_dir}/step{step}.h5"
        
        if os.path.isfile(pth):
            feature_dict = file_utils.get_features(
                pth, group_name="optimizer", keys=keys, out_keys=layers,
            )
            for layer in layers:
                weights[layer][f"step_{step}"] = feature_dict[layer]
    return weights


def get_default_plot_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_suffix",
        type=str,
        default="",
        help="extra image suffix",
        required=False,
    )
    # overwrite boolean flag: if True, ignores the fact that file already exists
    parser.add_argument(
        "--overwrite", 
        dest="overwrite", 
        action="store_true",
        default=False
    )
    # use_tex boolean flag: if set, will use tex rendering for matplotlib labels
    parser.add_argument(
        "--use_tex", 
        dest="use_tex", 
        action="store_true", 
        default=False
    )
    parser.add_argument(
        "--feats_path", 
        type=str, 
        help="feature path for PT checkpoints",
        default=None
    )
    parser.add_argument(
        "--model", 
        type=str, 
        help="model for PT checkpoints",
        default=None
    )
    parser.add_argument(
        "--anchor_freq",
        type=int,
        help="how many steps per anchor. Will probably differ with different batch sizes",
        default=5000,
    )
    parser.add_argument(
        "--stop", 
        type=int, 
        help="last timestep to consider",
        default=5000
    )

    return parser