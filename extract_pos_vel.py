import glob
import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
from utils import load
from utils import flags
from sklearn.decomposition import PCA


def main():
    exp_path = f"{ARGS.save_dir}/{ARGS.experiment}/{ARGS.expid}"
    step_names = glob.glob(f"{exp_path}/ckpt/*.tar")
    step_list = [int(s.split(".tar")[0].split("step")[1]) for s in step_names]
    #device = load.device(ARGS.gpu)
    device = "cpu"
    save_path = f"{exp_path}/feats"
    try:
        os.makedirs(save_path)
    except FileExistsError:
        if not ARGS.overwrite:
            print(
                "Feature directory exists and no-overwrite specified. Rerun with --overwrite"
            )
            quit()
    i = 0
    weight_traj = []
    vel_traj = []
    steps = []
    train_loss = []
    for in_filename, step in tqdm(
            sorted(list(zip(step_names, step_list)), key=lambda x: x[1])[1:]
    ):
        out_filename = f"{save_path}/step{step}.h5"

        if os.path.isfile(out_filename) and not ARGS.overwrite:
            print(f"\t{out_filename} already exists, skipping")
            continue
    
        checkpoint = torch.load(in_filename, map_location=device)
        if "position" in checkpoint.keys():
            metrics = {
                "position": checkpoint["position"],
                "velocity": checkpoint["velocity"],
                #"step": np.array(step),
                #"train_loss": checkpoint["train_loss"],
            }
            with h5py.File(out_filename, "w") as f:
                for k, v in metrics.items():
                    dset = f.create_dataset(k, v.shape, "f")
                    dset[:] = v

if __name__ == "__main__":
    parser = flags.extract()
    ARGS = parser.parse_args()
    main()
