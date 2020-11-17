import glob
import os
import deepdish as dd
import numpy as np
import torch
from tqdm import tqdm
from utils import load
from utils import flags


def main():
    exp_path = f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}"
    step_names = glob.glob(f"{exp_path}/ckpt/*.tar")
    step_list = [int(s.split(".tar")[0].split("step")[1]) for s in step_names]
    device = load.device(ARGS.gpu)

    save_path = f"{exp_path}/feats"
    try:
        os.makedirs(save_path)
    except FileExistsError:
        if not ARGS.overwrite:
            print(
                "Feature directory exists and no-overwrite specified. Rerun with --overwrite"
            )
            quit()

    for in_filename, step in tqdm(
        sorted(list(zip(step_names, step_list)), key=lambda x: x[1])
    ):
        out_filename = f"{save_path}/step{step}.h5"

        if os.path.isfile(out_filename) and not ARGS.overwrite:
            print(f"\t{out_filename} already exists, skipping")
            continue

        checkpoint = torch.load(in_filename, map_location=device)
        # Metrics
        metrics = {}
        for m in ["train_loss", "test_loss", "accuracy1", "accuracy5"]:
            if m in checkpoint.keys():
                metrics[m] = np.array([checkpoint[m]])
        # Weights
        params = {}
        for name, tensor in checkpoint["model_state_dict"].items():
            if "weight" in name:
                params[name] = tensor.cpu().numpy()
            if "bias" in name:
                params[name] = tensor.cpu().numpy()
        # Buffers
        buffers = {}
        # this assumes the same order of model state dict as optimize state dict
        param_names = [
            name
            for name in checkpoint["model_state_dict"].keys()
            if ("weight" in name or "bias" in name)
        ]
        for name, buffer_dict in zip(
            param_names, checkpoint["optimizer_state_dict"]["state"].values()
        ):
            if "weight" in name and "gradient_norm_buffer" in buffer_dict.keys():
                buffers[name] = buffer_dict["gradient_norm_buffer"].cpu().numpy()
            if "bias" in name and "gradient_norm_buffer" in buffer_dict.keys():
                buffers[name] = buffer_dict["gradient_norm_buffer"].cpu().numpy()
        dd.io.save(
            out_filename, {"metrics": metrics, "params": params, "buffers": buffers}
        )


if __name__ == "__main__":
    parser = flags.extract()
    ARGS = parser.parse_args()
    main()
