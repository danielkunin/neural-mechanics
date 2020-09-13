"""
Saves weights for PyTorch models
"""

import argparse
import glob
import os
import deepdish as dd
import torch
from tqdm import tqdm


def main():
    step_names = glob.glob(f"{FLAGS.ckpt_path}/ckpt/*.tar")
    step_list = [int(s.split(".tar")[0].split("step")[1]) for s in step_names]

    use_cuda = torch.cuda.is_available()
    device = torch.device(("cuda:" + str(FLAGS.gpu)) if use_cuda else "cpu")

    save_path = f"{FLAGS.ckpt_path}/feats"
    try:
        os.makedirs(save_path)
    except FileExistsError:
        if not FLAGS.overwrite:
            print(
                "Feature directory exists and no-overwrite specified. Rerun with --overwrite"
            )
            quit()

    for in_filename, step in tqdm(zip(step_names, step_list)):
        out_filename = f"{save_path}/step{step}.h5"

        if os.path.isfile(out_filename) and not FLAGS.overwrite:
            print(f"\t{out_filename} already exists, skipping")
            continue

        weights = {}
        biases = {}
        checkpoint = torch.load(in_filename, map_location=device)
        for name, tensor in checkpoint["model_state_dict"].items():
            if "weight" in name:
                weights[name] = tensor.cpu().numpy()
            if "bias" in name:
                biases[name] = tensor.cpu().numpy()

        dd.io.save(out_filename, {"weights": weights, "biases": biases})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="checkpoint path for PT checkpoints",
        required=True,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="Which gpu to use to load checkpoints to",
        default=0
    )
    parser.add_argument(
        "--overwrite", 
        dest="overwrite", 
        action="store_true",
        default=False
    )
    FLAGS, _ = parser.parse_known_args()
    main()
