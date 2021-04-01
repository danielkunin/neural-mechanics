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
        extracted_names = []
        for name, tensor in checkpoint["model_state_dict"].items():
            if "weight" in name or "bias" in name:
                params[name] = tensor.cpu().numpy()
                extracted_names.append(name)
        # Buffers
        buffers = {}
        if len(checkpoint["optimizer_state_dict"]["state"].keys()) > 0:
            # this assumes the same order of model state dict as optimize state dict
            # recall optimizer_state_dict.state has int keys, not strings
            all_param_names = [
                name
                for name in checkpoint["model_state_dict"].keys()
                if ("weight" in name or "bias" in name)
            ]
            assert len(all_param_names) == len(checkpoint["optimizer_state_dict"]["state"].keys())
            # Get the int keys for the names extracted above
            optimizer_keys = [
                (i, name)
                for i,name in enumerate(all_param_names)
                if name in extracted_names
            ]
            for opt_key,name in optimizer_keys:
                param_state = checkpoint["optimizer_state_dict"]["state"][opt_key]
                if "buffers" in param_state.keys():
                    buffer_dict = param_state["buffers"]
                    # Cannot nest dictionaries deeper: load function assumes only 2
                    # nested keys: one for the group, one for feat name
                    for k, v in buffer_dict.items():
                        buffers[f"{name}.{k}"] = v.cpu().numpy()
        dd.io.save(
            out_filename, {"metrics": metrics, "params": params, "buffers": buffers}
        )


if __name__ == "__main__":
    parser = flags.extract()
    ARGS = parser.parse_args()
    main()
