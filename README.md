# Self-supervised Representation Learning for Neuronal Morphologies

This repository is an adapted fork of [marissaweis/ssl_neuron](https://github.com/marissaweis/ssl_neuron)
(GraphDINO, from [Weis et al., 2023](https://openreview.net/forum?id=ThhMzfrd6r)), extended here to train on a
connectome skeleton dataset (`healed_syns_swc`) with node-type information as an extra feature.

We train **three GraphDINO models**, differing only in which part of the morphology they see:

| variant    | morphology                  | node types present                   |
|------------|-----------------------------|--------------------------------------|
| `full`     | whole skeleton              | soma, axon, dendrite, synapse        |
| `dendrite` | soma + dendrites only       | soma, dendrite, post-synaptic sites  |
| `axon`     | soma + axon only            | soma, axon, pre-synaptic boutons     |

All three use the 4-dim node-type one-hot (`feat_dim=7`), so the type channel stays
informative in every variant — within `dendrite` and `axon` it still separates
neurite from synapse.

> **Local fork changes vs. upstream:** a `use_type` feature mode (4-dim node-type one-hot →
> `feat_dim=7`), a compartment-aware SWC preprocessing package (`ssl_neuron/preprocessing/`),
> per-variant configs / checkpoints / latents, resume + override CLI flags in `main.py`, and a
> reusable latent-export script (`demos/save_latents.py`).


## Repository structure

```
ssl_neuron/                     # repo root
├── README.md                   # this file
├── run_pipeline.sh             # preprocess -> train -> latents, for all variants
├── environment.yml             # conda env spec (env name: "ssl")
└── ssl_neuron/                 # the Python package
    ├── main.py                 # training entry point (--variant full|dendrite|axon)
    ├── train.py                # Trainer: training loop, checkpointing, resume
    ├── graphdino.py            # GraphDINO model: GraphTransformer + DINO teacher/student
    ├── datasets.py             # GraphDataset: loads skeletons, augments graphs
    ├── utils.py                # graph ops (subsample, adjacency, Laplacian PE) + plotting
    │
    ├── preprocessing/          # SWC -> dataset (see preprocessing/README.md)
    │   ├── swc_io.py           # read/write SWC, neighbour maps, tree rebuilding
    │   ├── compartments.py     # compartment resolution + type mask with re-parenting
    │   ├── subsample.py        # stage 1 CLI: full SWC -> masked -> <=3000-node SWC
    │   ├── build_dataset.py    # stage 2 CLI: SWC -> features.npy/neighbors.pkl/splits
    │   ├── plot_swc.py         # 3D plotly QC plots of any SWC directory
    │   ├── variants.py         # the 3 variants and all data paths
    │   └── test_compartments.py
    │
    ├── configs/
    │   ├── full.json  dendrite.json  axon.json    # one per variant
    │   └── legacy/             # configs for the pre-reorganisation checkpoints
    │
    ├── demos/
    │   ├── save_latents.py     # latent export, per variant
    │   ├── compare_variants.py # 3-way comparison: t-SNE, cluster ARI, shared-kNN
    │   ├── load_data.ipynb  train_model.ipynb  model_inference_handcraft.ipynb
    │   └── embeddings/  figures/
    │
    ├── logs/                   # training logs (train_<variant>.log)
    └── archive/                # superseded notebooks + scripts (see archive/README.md)
```


## Environment

Conda env described by `environment.yml` (env name `ssl`):

```
conda env create -f environment.yml
conda activate ssl
```

> **numpy pin:** this project depends on compiled `scipy` (1.10.x) and `pandas` (1.5.x) built
> against numpy 1.x. Keep `numpy < 2` (tested with `numpy==1.26.4`); numpy 2.x breaks the
> scipy/pandas imports.

The package can also be installed in-place with `pip install -e .`. All commands below are run
**from the repo root** so that `ssl_neuron` is importable.


## Data

Source SWCs (one file per neuron, `neuron_<cell_id>_scale2_healed_syns.swc`, coordinates in EM
nanometres, `cell_id == segment_id`) live at `paths.json → skeletons_swc`. Everything this
pipeline generates goes under `<skeleton root>/graphdino/`:

```
/nfs/data8/chuyu/20230422_160839/connectome/skeletons/healed_syns_swc/
├── swc/                            # 467 source SWCs (input; unchanged)
└── graphdino/
    ├── swc_subsampled/<variant>/   # masked + subsampled SWCs (<=3000 nodes)
    ├── datasets/<variant>/         # skeletons/<cell_id>/{features.npy,neighbors.pkl}
    │                               # + all_ids.npy / train_ids.npy / val_ids.npy
    ├── ckpts/<variant>/            # ckpt_<epoch>.pt
    └── embeddings/                 # latents_<variant>.npz
```

`/nfs/roli8/data/chuyu/...` is the same directory via a different mount. Override the root with
the `SSL_NEURON_SKELETON_ROOT` environment variable.

**Feature layout.** `features.npy` is `(N, 8)`: columns `0:3` = soma-centred xyz in microns,
column `3` = radius (unused by the model), columns `4:8` = a 4-dim node-type one-hot
`[soma, axon, dendrite, synapse]`. `datasets.py` selects columns `[0,1,2]` when
`data.use_type` is false (feat_dim 3) or `[0,1,2,4,5,6,7]` when true (feat_dim 7). Checkpoints
are therefore **not** interchangeable across `use_type` settings — always pair a checkpoint with
the config it was trained with.

See [preprocessing/README.md](ssl_neuron/preprocessing/README.md) for how the compartment split
is defined and why it is done the way it is.


## Running the pipeline

```
./run_pipeline.sh preprocess    # SWC -> masked/subsampled SWC -> dataset, all variants
./run_pipeline.sh train         # one training run per variant, one per GPU
./run_pipeline.sh latents       # export latents for every variant
```

Or step by step:

```
python -m ssl_neuron.preprocessing.subsample      --variant all --jobs 16
python -m ssl_neuron.preprocessing.build_dataset  --variant all
CUDA_VISIBLE_DEVICES=0 python -m ssl_neuron.main --variant dendrite
python -m ssl_neuron.demos.save_latents --variant dendrite
```

Useful `main.py` overrides: `--resume PATH`, `--ckpt-dir DIR`, `--data-path DIR`,
`--max-iter N`, `--use-type / --no-use-type`.

**Checkpoint naming.** `train.py` saves every `save_ckpt_every` (500) **epochs**, as
`ckpt_<epoch>.pt` — not every 500 iterations. With ~6 iterations per epoch, the default
`max_iter` of 100000 is about 16.6k epochs.

**Latents.** `save_latents.py` writes `{latents (N, 32), cell_ids, checkpoint, variant}` per
variant, using the highest-numbered checkpoint unless `--ckpt` says otherwise. `cell_ids` are
segment ids, so variants with different cell counts are compared by intersecting ids.

**Comparing the variants.**

```
python -m ssl_neuron.demos.compare_variants --k 5
```

Restricts all three latent sets to their shared cells and writes `tsne_variants.png`,
`variant_agreement.png` (k-means adjusted Rand index + confusion matrices),
`latent_similarity.png` (shared k-nearest-neighbour fraction) and `compare_variants.json` to
`graphdino/figures/`.

**QC plots.**

```
python -m ssl_neuron.preprocessing.plot_swc --variant dendrite --limit 20
```

Writes one interactive 3D HTML per neuron, colour-coded by compartment, to
`graphdino/plots_html/<variant>/` — the quickest way to confirm the mask did what you expect.


## Cell counts

The compartment mask drops cells that have (almost) nothing left in a compartment — in
particular the 12 reconstructions that carry no compartment labels at all. The validation split
is drawn once from the source list and intersected per variant, so the same neurons are held out
everywhere.

| variant    | cells | train | val |
|------------|-------|-------|-----|
| `full`     |  464  |  414  |  50 |
| `dendrite` |  455  |  408  |  47 |
| `axon`     |  454  |  407  |  47 |

Three source cells are dropped from `full` as well: their soma sits in a 16–93 node fragment of a
badly broken reconstruction (see `preprocessing/README.md`).


## Moving to another server — checklist

- Recreate the env: `conda env create -f environment.yml` (keep `numpy<2`).
- The datasets, checkpoints and latents all live on NFS under `graphdino/`, so only the repo
  needs copying — but set `SSL_NEURON_SKELETON_ROOT` if that mount is elsewhere, and regenerate
  `configs/*.json` (they store absolute paths).
- `ckpts/`, `*.pt` and `data/skeletons/` are git-ignored and are not in the git history.


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
