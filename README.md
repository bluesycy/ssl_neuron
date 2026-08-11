# Self-supervised Representation Learning for Neuronal Morphologies

This repository is an adapted fork of [marissaweis/ssl_neuron](https://github.com/marissaweis/ssl_neuron)
(GraphDINO, from [Weis et al., 2023](https://openreview.net/forum?id=ThhMzfrd6r)), extended here to train on a
connectome skeleton dataset (`healed_syns_swc`) and to optionally use node-type information as an extra feature.

> **Local fork changes vs. upstream:** an optional `use_type` feature mode (4-dim node-type one-hot →
> `feat_dim=7`), a second checkpoint directory (`ckpts_with_type/`), extra inference/comparison notebooks,
> SWC preprocessing + plotting utilities, resume/override CLI flags in `main.py`, and a reusable latent-export
> script (`demos/save_latents.py`).


## Repository structure

```
ssl_neuron/                     # repo root
├── README.md                   # this file
├── LICENSE                     # MIT
├── setup.py / pyproject.toml   # package metadata (installs the `ssl_neuron` package)
├── environment.yml             # conda env spec (env name: "ssl")
└── ssl_neuron/                 # the Python package
    ├── main.py                 # training entry point (config + CLI overrides)
    ├── train.py                # Trainer: training loop, checkpointing, resume
    ├── graphdino.py            # GraphDINO model: GraphTransformer + DINO teacher/student
    ├── datasets.py             # GraphDataset: loads skeletons, augments graphs
    ├── utils.py                # graph ops (subsample, adjacency, Laplacian PE) + plotting helpers
    ├── plot_healed.py          # plot full SWC neurons (plotly/networkx)
    ├── plot_healed_subsample.py# subsample SWC graphs and plot
    │
    ├── configs/
    │   ├── config.json         # train WITHOUT type info  (feat_dim=3, ckpt_dir=ckpts/)
    │   ├── config_with_type.json # train WITH type info   (feat_dim=7, ckpt_dir=ckpts_with_type/)
    │   └── config_inf.json      # inference config used by the demo notebooks
    │
    ├── ckpts/                  # checkpoints, xyz-only model   (git-ignored; ckpt_*.pt every 500 iters)
    ├── ckpts_with_type/        # checkpoints, xyz+type model    (git-ignored)
    │
    ├── data/                   # dataset root (see "Data" below)
    │   ├── skeletons/<cell_id>/features.npy   # (N, 8): xyz, radius, 4-dim type one-hot   (git-ignored)
    │   ├── skeletons/<cell_id>/neighbors.pkl  # dict: node_id -> set(neighbor ids)
    │   ├── all_ids.npy / train_ids.npy / val_ids.npy  # cell-id splits (466 / 416 / 50)
    │   ├── data_utils.py
    │   ├── extract_allen_data.ipynb           # upstream Allen preprocessing
    │   ├── save_skeleton_features.ipynb       # SWC -> features.npy + neighbors.pkl
    │   └── save_skeleton_features_without_axon.ipynb
    │
    ├── demos/
    │   ├── load_data.ipynb
    │   ├── train_model.ipynb
    │   ├── model_inference.ipynb              # inference, no type    (uses config_inf.json)
    │   ├── model_inference_with_type.ipynb    # inference, with type  (uses config_with_type.json)
    │   ├── model_inference_compare.ipynb      # compares both variants
    │   ├── model_inference_handcraft.ipynb
    │   ├── save_latents.py                    # reusable latent export (see "Inference")
    │   ├── embeddings/                        # saved latent arrays
    │   └── figures/                           # generated plots (t-SNE, clusters, comparisons)
    │
    └── logs/                   # training logs (train_*.log)
```

**Feature layout.** Saved `features.npy` is `(N, 8)`: columns `0:3` = xyz, col `3` = radius (unused),
cols `4:8` = 4-dim node-type one-hot. `datasets.py` selects columns based on `data.use_type`:
`[0,1,2]` when `false` (feat_dim 3) or `[0,1,2,4,5,6,7]` when `true` (feat_dim 7). The two checkpoint
directories therefore correspond to models with different input dimensionality and are **not**
interchangeable — always pair a checkpoint with the config it was trained with.


## Environment

The code runs in the conda env described by `environment.yml` (env name `ssl`):

```
conda env create -f environment.yml
conda activate ssl
```

> **numpy pin:** this project depends on compiled `scipy` (1.10.x) and `pandas` (1.5.x) built against
> numpy 1.x. Keep `numpy < 2` (tested with `numpy==1.26.4`); numpy 2.x breaks scipy/pandas imports.

The package can also be installed in-place with `pip install -e .` (or `python3 setup.py install`).


## Data

Each neuron is a soma-centered graph stored under `data/skeletons/<cell_id>/` as `features.npy` and
`neighbors.pkl`. Splits are given by `all_ids.npy`, `train_ids.npy`, `val_ids.npy`.

- Upstream ABA/Allen preprocessing: `data/extract_allen_data.ipynb` (see `data/README.md`).
- This fork's connectome pipeline: `data/save_skeleton_features.ipynb` converts the source SWC files
  (`neuron_<cell_id>_scale2_healed_syns.swc`) into `features.npy` + `neighbors.pkl`.

To use a custom dataset, point `data.path` in the config to a directory laid out as above. See
[data/README.md](ssl_neuron/data/README.md) for the full format spec.


## Training

Train without type info (xyz only):
```
python3 ssl_neuron/main.py --config=ssl_neuron/configs/config.json
```

Train with type info (xyz + node type):
```
python3 ssl_neuron/main.py --config=ssl_neuron/configs/config_with_type.json
```

Useful CLI overrides (see `main.py`):
- `--resume PATH` — continue from a checkpoint.
- `--use-type / --no-use-type` — force `feat_dim` on/off regardless of config.
- `--ckpt-dir DIR` — write checkpoints to a separate dir (keeps old ckpts intact).

Checkpoints are written to `trainer.ckpt_dir` every `save_ckpt_every` (500) iterations as `ckpt_<iter>.pt`,
each containing `{model, optimizer, epoch, curr_iter, loss}`.


## Inference / exporting latents

Demo notebooks in `demos/` show loading, plotting, clustering, and t-SNE.

For batch export, use the reusable script (run from the repo root so `ssl_neuron` is importable):
```
python -m ssl_neuron.demos.save_latents
```
By default it runs both variants, auto-selecting the highest-iteration checkpoint in each of
`ckpts/` and `ckpts_with_type/`, and writes `latents_no_type.npz` / `latents_with_type.npz`
(each `{latents (N,32), cell_ids, checkpoint}`) to the `--out-dir`. Override a single run with
`--config/--ckpt/--out`; override the dataset with `--data-path`.


## Checkpoints & backup

`ckpts/`, `ckpts_with_type/`, `*.pt`, and `data/skeletons/` are **git-ignored** (see `.gitignore`), so they
are **not** in the git history and must be copied separately when moving the repo.

A backup of the trained checkpoints, the source SWCs, and the exported latents lives at:
```
/nfs/roli8/data/chuyu/20230422_160839/connectome/skeletons/healed_syns_swc/
├── swc/                 # source SWC skeletons (neuron_<id>_scale2_healed_syns.swc)
├── embeddings/          # latents_no_type.npz, latents_with_type.npz
├── ckpts/               # backup of xyz-only checkpoints
└── ckpts_with_type/     # backup of xyz+type checkpoints
```


## Moving to another server — checklist

- Recreate the env: `conda env create -f environment.yml` (keep `numpy<2`).
- Copy the git-ignored assets that are not in the repo: `ckpts/`, `ckpts_with_type/`, and
  `data/skeletons/` (restore from the NFS backup above).
- Fix hard-coded paths that assume this machine:
  - `configs/config_inf.json` uses `../data/`; the training configs use `data/` (relative to the run dir).
  - `demos/save_latents.py` defaults `--out-dir` to the NFS path above — pass `--out-dir`/`--data-path`
    on the new host.
  - The demo notebooks reference `../ckpts/` and `../data/` relative to `demos/`.


## Citation

If you use this repository in your research, please cite the original GraphDINO paper:
```
@article{Weis2023,
      title={Self-Supervised Graph Representation Learning for Neuronal Morphologies},
      author={Marissa A. Weis and Laura Hansel and Timo L{\"u}ddecke and Alexander S. Ecker},
      journal={Transactions on Machine Learning Research},
      issn={2835-8856},
      year={2023}
}
```


## License
This project is covered under the MIT License.
