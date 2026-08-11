# Archive

Superseded code, kept for reference. **None of it is part of the current pipeline** and the paths
inside it are stale.

## `scripts/`

| file                        | superseded by |
|-----------------------------|---------------|
| `plot_healed.py`            | `preprocessing/plot_swc.py` |
| `plot_healed_subsample.py`  | `preprocessing/subsample.py` + `preprocessing/plot_swc.py` |

`plot_healed_subsample.prune_axon_nodes()` removed axon **subtrees**. That is the wrong operation
for these skeletons: their parent pointers are not compartment-aware, so subtree deletion also
discards ~18% of genuine dendrite. See `preprocessing/README.md`.

## `notebooks/`

| file                                     | superseded by |
|------------------------------------------|---------------|
| `save_skeleton_features.ipynb`           | `preprocessing/build_dataset.py --variant full` |
| `save_skeleton_features_without_axon.ipynb` | `preprocessing/build_dataset.py --variant dendrite` |
| `model_inference.ipynb`                  | `demos/save_latents.py` |
| `model_inference_with_type.ipynb`        | `demos/save_latents.py` |
| `model_inference_compare.ipynb`          | `demos/compare_variants.py` |
| `extract_allen_data.ipynb`               | upstream Allen preprocessing; never used for the connectome data |

The two `save_skeleton_features` notebooks also folded SWC type 0 (undefined) into the dendrite
one-hot slot; the current pipeline resolves it from the node's compartment instead.

## `*_ids.npy`

The id splits from the pre-reorganisation dataset (466 / 416 / 50 cells). Current splits live per
variant under `<skeleton root>/graphdino/datasets/<variant>/`.
