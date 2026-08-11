# `data/` — dataset format and the upstream Allen pipeline

The connectome pipeline that this fork actually uses lives in
[`../preprocessing/`](../preprocessing/README.md); its outputs go to NFS under
`<skeleton root>/graphdino/datasets/<variant>/`, not into this directory. What
remains here is `data_utils.py` (helpers shared with the upstream code) and this
format spec.

The upstream Allen preprocessing notebooks and the superseded connectome notebooks are in
[`../archive/notebooks/`](../archive/notebooks/).

## Dataset format

`config.data.path` must point at a directory laid out as:

```
<data.path>/
├── skeletons/<cell_id>/features.npy    # (N, 8) float
├── skeletons/<cell_id>/neighbors.pkl   # {node_idx: set(neighbour idx)}
├── all_ids.npy
├── train_ids.npy
└── val_ids.npy
```

- **features.npy** — columns `0:3` soma-centred xyz in microns, column `3` radius (unused by the
  model), columns `4:8` a `[soma, axon, dendrite, synapse]` one-hot. `datasets.py` reads
  `[0,1,2]` when `data.use_type` is false and `[0,1,2,4,5,6,7]` when true.
- **neighbors.pkl** — undirected adjacency over node indices `0..N-1`, where index `0` is the
  soma. `datasets.py` hard-codes `soma_id = 0`.
- Graphs must be a single connected component; `datasets.py` does not repair them.

## Upstream ABA / Allen data

To download the ABA dataset, use the
[Allen Software Development Kit](http://alleninstitute.github.io/AllenSDK/cell_types.html). See the
[demo notebook](http://alleninstitute.github.io/AllenSDK/_static/examples/nb/cell_types.html#Cell-Morphology-Reconstructions)
on how to use the Allen Cell Types Database.

To get the rotation angles, download Dataset 3 from the Supplementary material of
[Gouwens et al. (2019)](https://www.nature.com/articles/s41593-019-0417-0#Sec27); the column
`upright_angle` contains the information to rotate the cell to vertical.

`../archive/notebooks/extract_allen_data.ipynb` has the upstream preprocessing.

### Pretrained upstream model

The upstream pretrained checkpoint was trained after removing the axons and centering each neuron
so the soma is at `(0, 0, 0)`, using only xyz coordinates as node features (`feat_dim=3`).
