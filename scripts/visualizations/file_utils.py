"""
Utilities for dealing with file I/O
"""

import os
import pprint
import h5py
from scripts.visualizations import utils  


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

    keys = utils.make_iterable(keys)

    if out_keys is None:
        out_keys = keys
    out_keys = utils.make_iterable(out_keys)

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
