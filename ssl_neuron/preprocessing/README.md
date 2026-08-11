# Preprocessing: SWC → GraphDINO dataset

Two stages, each a CLI:

```
python -m ssl_neuron.preprocessing.subsample      --variant all --jobs 16
python -m ssl_neuron.preprocessing.build_dataset  --variant all
```

```
swc/neuron_<id>_scale2_healed_syns.swc          full reconstruction, ~28k nodes median
   │
   │  stage 1 (subsample.py)
   │    1. resolve every node's compartment          compartments.resolve_compartments
   │    2. keep the variant's compartments,          compartments.mask_compartments
   │       re-parenting survivors
   │    3. restrict to the soma's component          compartments.largest_component
   │    4. contract to <=3000 nodes, soma protected  utils.subsample_graph
   ▼
graphdino/swc_subsampled/<variant>/*.swc
   │
   │  stage 2 (build_dataset.py)
   │    5. renumber ids 0..N-1 (row 0 = soma), nm -> microns, soma -> origin
   │    6. write features.npy (N,8) + neighbors.pkl, and the id splits
   ▼
graphdino/datasets/<variant>/skeletons/<cell_id>/
```

## SWC type convention

This dataset does **not** follow the stock SWC spec. Per the collaborator:

```
0 = undefined    1 = soma    2 = axon    3 = dendrite    7 = synapse
```

Row 0 of every file is the soma; coordinates are EM nanometres.

## Compartment assignment

A node's compartment is its own type when that is soma/axon/dendrite. Synapse (7) and undefined
(0) nodes inherit from their nearest **axon or dendrite** ancestor, so post-synaptic sites follow
the dendrite and pre-synaptic boutons follow the axon.

The soma deliberately does **not** propagate its label downward. A node that reaches the soma
without passing through a labelled neurite is genuinely unassigned and is dropped from the
compartment-specific variants. Without this rule the 12 reconstructions that are >50% type-0 —
they carry no compartment labels at all — would survive *whole* in both the dendrite and the axon
variant and contaminate the comparison.

The `type` column itself is never rewritten, so the saved one-hot still distinguishes synapse
nodes inside every variant. Nodes that resolve to no compartment keep an **all-zero** type vector
rather than being assigned a category they do not belong to (about 31k nodes in `full`, almost
all of them in those 12 unlabelled cells; 2–4 nodes total across the compartment variants).

## Why mask, rather than delete subtrees

The parent pointers in these SWCs are **not** compartment-aware — they are closer to a geometric
spanning tree:

- 18.5% of dendrite nodes sit downstream of an axon node
- 92% of axon nodes sit downstream of a dendrite node

So deleting whole subtrees (what the archived `plot_healed_subsample.prune_axon_nodes` did)
throws away roughly 18% of genuine dendrite. Masking by node type and re-parenting each survivor
to its nearest surviving ancestor is the correct operation, and it barely fragments the tree:
masking the full SWCs leaves a median of 1–2 orphan components per cell (max 36 over 466 cells),
which re-parenting stitches straight back on. Median re-parent jump is ~1 µm for the dendrite
variant and ~18 µm for the axon variant (the axon initial segment reconnecting to the soma).

## Why mask before subsampling

Masking the *already subsampled* 3000-node files would starve the axon variant: 194 of 466 cells
would end up under 1000 nodes and 23 under 200. Masking the full SWC first and then spending the
whole 3000-node budget on the surviving compartment leaves only a handful of cells short.

## Disconnected source reconstructions

Some source SWCs are fragmented — the worst has 474 components. `largest_component` keeps only
the component containing the soma, which is what GraphDINO assumes (a soma-rooted connected
tree). 459 of 467 cells still have ≥3000 nodes in the soma's component; 12 lose more than 20%,
and 3 are dropped entirely because their soma sits in a 16–93 node fragment.

The alternative — `data_utils.connect_graph`, which stitches components by nearest-neighbour
distance — fabricates edges that were never reconstructed, so it is not used here.

## Splits

`val_ids` are drawn once (50 cells, seed 42) from the full source list and then intersected with
each variant's surviving cells, so the same neurons are held out in all three variants.

## Tests

```
python -m pytest ssl_neuron/preprocessing/test_compartments.py
```

Covers compartment inheritance, the soma-no-propagation rule, re-parenting to the nearest
surviving ancestor, one-hot slots, and the unlabelled-cell case.
