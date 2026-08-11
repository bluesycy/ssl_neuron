"""Compute GraphDINO latents for the neuron dataset and save them to disk.

This reproduces the inference loop from ``model_inference.ipynb`` in a
reusable script so latents can be regenerated for future checkpoints.

By default it runs both trained variants and writes the resulting latent
matrices next to the source SWC data:

  * without type info  -> checkpoint in ``ckpts/``            (feat_dim=3)
  * with    type info  -> checkpoint in ``ckpts_with_type/``  (feat_dim=7)

Run from the repository root so that the ``ssl_neuron`` package is importable::

    python -m ssl_neuron.demos.save_latents

Or point it at a specific config/checkpoint/output::

    python -m ssl_neuron.demos.save_latents \
        --config ssl_neuron/configs/config_with_type.json \
        --ckpt   ssl_neuron/ckpts_with_type/ckpt_16500.pt \
        --out    /path/to/latents_with_type.npz
"""

import os
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ssl_neuron.datasets import GraphDataset
from ssl_neuron.utils import (
    neighbors_to_adjacency_torch,
    compute_eig_lapl_torch_batch,
)
from ssl_neuron.graphdino import create_model


# Repository root (…/ssl_neuron) and the inner package dir (…/ssl_neuron/ssl_neuron).
PKG_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_DIR.parent

DEFAULT_DATA_PATH = PKG_DIR / "data"
DEFAULT_OUT_DIR = Path(
    "/nfs/roli8/data/chuyu/20230422_160839/connectome/skeletons/healed_syns_swc/embeddings"
)

# (config, checkpoint dir, output basename) for the two trained variants.
VARIANTS = [
    ("config.json", "ckpts", "latents_no_type"),
    ("config_with_type.json", "ckpts_with_type", "latents_with_type"),
]


def latest_checkpoint(ckpt_dir):
    """Return the checkpoint file with the highest iteration number."""
    ckpts = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(ckpts, key=lambda p: int(Path(p).stem.split("_")[1]))


def compute_latents(config, ckpt_path, device="cuda"):
    """Run the encoder over every neuron in the dataset and return latents + ids."""
    model = create_model(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)

    # ``mode='all'`` loads every cell listed in data/all_ids.npy.
    dset = GraphDataset(config, mode="all")

    dim = config["model"]["dim"]
    pos_dim = config["model"]["pos_dim"]
    latents = np.zeros((dset.num_samples, dim), dtype=np.float32)
    cell_ids = np.zeros(dset.num_samples, dtype=np.int64)

    with torch.no_grad():
        for i in tqdm(range(dset.num_samples), desc=Path(ckpt_path).parent.name):
            feat, neigh = dset.__getsingleitem__(i)
            cell_ids[i] = int(dset.cells[i]["cell_id"])

            adj = neighbors_to_adjacency_torch(
                neigh, list(neigh.keys())
            ).float().to(device)[None, ]
            lapl = compute_eig_lapl_torch_batch(
                adj, pos_enc_dim=pos_dim
            ).float().to(device)
            feat = torch.from_numpy(feat).float().to(device)[None, ]

            latents[i] = model.student_encoder.forward(feat, adj, lapl)[0].cpu().numpy()

    return latents, cell_ids


def run_variant(config_name, ckpt_dir_name, out_basename, args):
    config_path = PKG_DIR / "configs" / config_name
    config = json.load(open(config_path))
    # Always resolve the dataset to an absolute path so the script is
    # runnable from any working directory.
    config["data"]["path"] = str(args.data_path)

    ckpt_dir = PKG_DIR / ckpt_dir_name
    ckpt_path = args.ckpt or latest_checkpoint(str(ckpt_dir))

    print(f"\n=== {out_basename} ===")
    print(f"config:     {config_path}")
    print(f"checkpoint: {ckpt_path}")
    print(f"use_type:   {config['data'].get('use_type', False)}  (feat_dim={config['data']['feat_dim']})")

    latents, cell_ids = compute_latents(config, ckpt_path, device=args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_basename}.npz"
    np.savez(
        out_path,
        latents=latents,
        cell_ids=cell_ids,
        checkpoint=os.path.basename(ckpt_path),
    )
    print(f"saved {latents.shape} latents -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH),
                        help="Directory holding all_ids.npy and skeletons/.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="Directory to write the latent .npz files into.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Single-variant overrides (optional).
    parser.add_argument("--config", default=None, help="Path to a single config to run.")
    parser.add_argument("--ckpt", default=None, help="Path to a single checkpoint to run.")
    parser.add_argument("--out", default=None, help="Output .npz path for the single-variant run.")
    args = parser.parse_args()

    if args.config:
        # Single explicit run.
        config = json.load(open(args.config))
        config["data"]["path"] = str(args.data_path)
        ckpt_path = args.ckpt or latest_checkpoint(str(PKG_DIR / config["trainer"]["ckpt_dir"]))
        latents, cell_ids = compute_latents(config, ckpt_path, device=args.device)
        out_path = Path(args.out or (Path(args.out_dir) / "latents.npz"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, latents=latents, cell_ids=cell_ids,
                 checkpoint=os.path.basename(ckpt_path))
        print(f"saved {latents.shape} latents -> {out_path}")
        return

    for config_name, ckpt_dir_name, out_basename in VARIANTS:
        run_variant(config_name, ckpt_dir_name, out_basename, args)


if __name__ == "__main__":
    main()
