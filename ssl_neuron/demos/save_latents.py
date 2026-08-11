"""Compute GraphDINO latents for each compartment variant and save them to disk.

Runs the encoder over every cell in a variant's `all_ids.npy` and writes
`{latents (N, 32), cell_ids, checkpoint, variant}` to
`<work_root>/embeddings/latents_<variant>.npz`.

Run from the repository root so that the `ssl_neuron` package is importable::

    python -m ssl_neuron.demos.save_latents                    # all variants
    python -m ssl_neuron.demos.save_latents --variant dendrite

By default the highest-iteration checkpoint in the variant's `ckpt_dir` is used;
override with `--ckpt`. Latents are keyed by `cell_ids` (== segment_id), so
variants with different cell counts can be compared by intersecting the ids.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ssl_neuron.datasets import GraphDataset
from ssl_neuron.graphdino import create_model
from ssl_neuron.preprocessing.variants import VARIANTS, get_variant
from ssl_neuron.utils import compute_eig_lapl_torch_batch, neighbors_to_adjacency_torch

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'configs'


def latest_checkpoint(ckpt_dir):
    """Return the checkpoint file with the highest iteration number."""
    ckpts = glob.glob(os.path.join(str(ckpt_dir), 'ckpt_*.pt'))
    if not ckpts:
        raise FileNotFoundError(f'No checkpoints found in {ckpt_dir}')
    return max(ckpts, key=lambda p: int(Path(p).stem.split('_')[1]))


def compute_latents(config, ckpt_path, device='cuda', desc=''):
    """Run the encoder over every neuron in the dataset and return latents + ids."""
    model = create_model(config)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    # `mode='all'` loads every cell listed in <data.path>/all_ids.npy.
    dset = GraphDataset(config, mode='all', inference=True)

    dim = config['model']['dim']
    pos_dim = config['model']['pos_dim']
    latents = np.zeros((dset.num_samples, dim), dtype=np.float32)
    cell_ids = np.zeros(dset.num_samples, dtype=np.int64)

    with torch.no_grad():
        for i in tqdm(range(dset.num_samples), desc=desc or Path(ckpt_path).parent.name):
            feat, neigh = dset.__getsingleitem__(i)
            cell_ids[i] = int(dset.cells[i]['cell_id'])

            adj = neighbors_to_adjacency_torch(
                neigh, list(neigh.keys())).float().to(device)[None, ]
            lapl = compute_eig_lapl_torch_batch(adj, pos_enc_dim=pos_dim).float().to(device)
            feat = torch.from_numpy(feat).float().to(device)[None, ]

            latents[i] = model.student_encoder.forward(feat, adj, lapl)[0].cpu().numpy()

    return latents, cell_ids


def run_variant(name, args):
    variant = get_variant(name)
    config = json.load(open(CONFIG_DIR / f'{name}.json'))
    if args.data_path:
        config['data']['path'] = args.data_path

    ckpt_path = args.ckpt or latest_checkpoint(config['trainer']['ckpt_dir'])
    out_path = Path(args.out) if args.out else variant.latents_path

    print(f'\n=== {name} ===')
    print(f'dataset:    {config["data"]["path"]}')
    print(f'checkpoint: {ckpt_path}')
    print(f'use_type:   {config["data"].get("use_type", False)} '
          f'(feat_dim={config["data"]["feat_dim"]})')

    latents, cell_ids = compute_latents(config, ckpt_path, device=args.device, desc=name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, latents=latents, cell_ids=cell_ids,
             checkpoint=os.path.basename(ckpt_path), variant=name)
    print(f'saved {latents.shape} latents -> {out_path}')
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', default='all',
                        help=f'one of {sorted(VARIANTS)} or "all"')
    parser.add_argument('--ckpt', default=None,
                        help='explicit checkpoint (single-variant runs only)')
    parser.add_argument('--out', default=None,
                        help='explicit output .npz (single-variant runs only)')
    parser.add_argument('--data-path', default=None, help='override data.path')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    names = sorted(VARIANTS) if args.variant == 'all' else [args.variant]
    if len(names) > 1 and (args.ckpt or args.out):
        raise SystemExit('--ckpt/--out only make sense with a single --variant')

    for name in names:
        run_variant(name, args)


if __name__ == '__main__':
    main()
