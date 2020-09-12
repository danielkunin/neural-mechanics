"""
Generic utility functions
"""
import importlib
import os
import sys

import numpy as np

# # find the git root
# git_repo = git.Repo(".", search_parent_directories=True)
# git_root = git_repo.git.rev_parse("--show-toplevel")
# config_path = "%s/configs" % git_root
# sys.path.append(config_path)


def string_to_list(s, delimiter=",", casting_fn=str):
    """ Casts (with no checks) each part of a delimited string a member of a list"""
    return [casting_fn(part) for part in s.split(delimiter)]


def makedir_quiet(d):
    """
    Convenience util to create a directory if it doesn't exist
    """
    if not os.path.isdir(d):
        os.makedirs(d)


def make_iterable(x):
    """
    If x is not already array_like, turn it into a list or np.array

    Inputs
        x: either array_like (in which case nothing happens) or non-iterable,
            in which case it gets wrapped by a list
    """

    if not isinstance(x, (list, tuple, np.ndarray)):
        return [x]
    return x


def get_config(exp_id):
    try:
        config_module = importlib.import_module(exp_id.replace("-", "_"))
        config = config_module.get_config()
        return config
    except ImportError as ie:
        print(
            "Module %s does not exist! Add it to configs/ or check your spelling."
            % exp_id
        )
        raise ie
