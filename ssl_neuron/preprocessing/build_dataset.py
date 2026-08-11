"""Stage 2: subsampled SWC -> GraphDINO dataset.

Writes, under `<variant>.dataset_dir`:

    skeletons/<cell_id>/features.npy   (N, 8) float64
    skeletons/<cell_id>/neighbors.pkl  {node_idx: set(neighbour idx)}
    all_ids.npy / train_ids.npy / val_ids.npy

Feature layout (unchanged from the original notebook, so old checkpoints stay
loadable): columns 0:3 = soma-centred xyz in microns, column 3 = radius,
columns 4:8 = 4-dim node-type one-hot [soma, axon, dendrite, synapse].
`datasets.py` reads columns [0,1,2] or [0,1,2,4,5,6,7] depending on
`data.use_type`.

Undefined (type 0) nodes are one-hot encoded as their resolved compartment
rather than being lumped into the dendrite slot, so an unlabelled node on the
axon reads as axon.

Usage:

    python -m ssl_neuron.preprocessing.build_dataset --variant all
"""

import argparse
import pickle
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

from ssl_neuron.preprocessing import compartments as comp_mod
from ssl_neuron.preprocessing import swc_io
from ssl_neuron.preprocessing.swc_io import AXON, DENDRITE, SOMA, SYNAPSE, UNDEFINED
from ssl_neuron.preprocessing.variants import (
    MIN_NODES,
    SOURCE_SWC_DIR,
    VARIANTS,
    get_variant,
)

#: One-hot slot per SWC type. Undefined (0) is resolved via its compartment.
TYPE_TO_SLOT = {SOMA: 0, AXON: 1, DENDRITE: 2, SYNAPSE: 3}

#: Node coordinates are EM nanometres; GraphDINO expects microns.
DOWNSCALE = 1000.0

#: Number of validation cells, drawn once and shared across variants.
N_VAL = 50
SPLIT_SEED = 42


def one_hot_slots(df):
    """Slot index per row for the node-type one-hot; -1 means "leave all zeros".

    Undefined (type 0) nodes take the slot of their resolved compartment, so an
    unlabelled node on the axon reads as axon instead of being lumped into the
    dendrite slot. Nodes that resolve to no compartment at all keep an all-zero
    type vector rather than being assigned a category they do not belong to.
    """
    types = df['type'].to_numpy(dtype=int)
    unknown = set(types.tolist()) - set(TYPE_TO_SLOT) - {UNDEFINED}
    assert not unknown, f'unexpected SWC type values: {unknown}'

    slots = np.array([TYPE_TO_SLOT.get(int(t), -1) for t in types], dtype=int)

    undefined = slots == -1
    if undefined.any():
        comp = comp_mod.resolve_compartments(df)
        slots[undefined] = [TYPE_TO_SLOT.get(int(c), -1) for c in comp[undefined]]
    return slots


def is_connected(neighbors, root=0):
    """BFS reachability check — cheaper than building an N x N adjacency."""
    seen = {root}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for neighbor in neighbors[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == len(neighbors)


def build_one(path):
    """Turn one subsampled SWC into (features, neighbors).

    Node ids are renumbered to 0..N-1 in file order, so index 0 is the soma —
    which is what `datasets.py` assumes (`soma_id = 0`).
    """
    df = swc_io.read_swc(path)
    df = swc_io.renumber(df, correction=True)

    slots = one_hot_slots(df)

    n = len(df)
    features = np.zeros((n, 8), dtype=float)
    features[:, :3] = df[['x', 'y', 'z']].to_numpy(dtype=float) / DOWNSCALE
    features[:, 3] = df['r'].to_numpy(dtype=float) / DOWNSCALE
    labelled = slots >= 0
    features[np.arange(n)[labelled], 4 + slots[labelled]] = 1.0

    # Soma-centre the graph (row 0 is the soma after renumbering).
    features[:, :3] -= features[0, :3]

    neighbors = {i: set() for i in range(n)}
    ids = df['id'].to_numpy(dtype=int)
    parents = df['parent'].to_numpy(dtype=int)
    for node_id, parent_id in zip(ids, parents):
        if parent_id >= 0:
            neighbors[int(node_id)].add(int(parent_id))
            neighbors[int(parent_id)].add(int(node_id))

    assert not np.any(np.isnan(features)), f'NaNs in features for {path}'
    assert is_connected(neighbors), f'{path} is not a single connected component'
    return features, neighbors


def choose_val_ids(source_dir, n_val=N_VAL, seed=SPLIT_SEED):
    """Pick the validation cells once, from the full source list.

    Drawing from the source (rather than per variant) keeps the same neurons
    held out in all three variants, so the models stay comparable.
    """
    all_source_ids = sorted(
        swc_io.cell_id_from_path(p) for p in swc_io.list_swc_files(source_dir)
    )
    return set(random.Random(seed).sample(all_source_ids, n_val))


def run_variant(variant, source_dir, min_nodes=MIN_NODES):
    swc_paths = swc_io.list_swc_files(variant.swc_dir)
    if not swc_paths:
        raise SystemExit(
            f'[{variant.name}] no SWCs in {variant.swc_dir} — '
            'run `python -m ssl_neuron.preprocessing.subsample` first'
        )

    dataset_dir = variant.dataset_dir
    skeleton_dir = dataset_dir / 'skeletons'
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    print(f'[{variant.name}] {len(swc_paths)} SWCs -> {dataset_dir}', flush=True)

    kept_ids, sizes, skipped = [], [], []
    for i, path in enumerate(swc_paths, 1):
        cell_id = swc_io.cell_id_from_path(path)
        features, neighbors = build_one(path)

        if len(features) < min_nodes:
            skipped.append((cell_id, len(features)))
            continue

        cell_dir = skeleton_dir / str(cell_id)
        cell_dir.mkdir(parents=True, exist_ok=True)
        np.save(cell_dir / 'features.npy', features)
        with open(cell_dir / 'neighbors.pkl', 'wb') as f:
            pickle.dump(neighbors, f, pickle.HIGHEST_PROTOCOL)

        kept_ids.append(cell_id)
        sizes.append(len(features))
        if i % 50 == 0 or i == len(swc_paths):
            print(f'[{variant.name}] {i}/{len(swc_paths)}', flush=True)

    val_pool = choose_val_ids(source_dir)
    val_ids = sorted(c for c in kept_ids if c in val_pool)
    train_ids = sorted(c for c in kept_ids if c not in val_pool)

    np.save(dataset_dir / 'all_ids.npy', np.array(sorted(kept_ids), dtype=np.int64))
    np.save(dataset_dir / 'train_ids.npy', np.array(train_ids, dtype=np.int64))
    np.save(dataset_dir / 'val_ids.npy', np.array(val_ids, dtype=np.int64))

    sizes = np.array(sizes) if sizes else np.array([0])
    print(f'[{variant.name}] {len(kept_ids)} cells '
          f'(nodes: min={sizes.min()} median={int(np.median(sizes))} max={sizes.max()}); '
          f'{len(train_ids)} train / {len(val_ids)} val; '
          f'{len(skipped)} skipped below {min_nodes} nodes')
    for cell_id, n in skipped:
        print(f'    skipped {cell_id}: {n} nodes', file=sys.stderr)
    return kept_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', default='all',
                        help=f'one of {sorted(VARIANTS)} or "all"')
    parser.add_argument('--source-dir', default=str(SOURCE_SWC_DIR),
                        help='full source SWCs, used only to draw the shared val split')
    parser.add_argument('--min-nodes', type=int, default=MIN_NODES)
    args = parser.parse_args()

    names = sorted(VARIANTS) if args.variant == 'all' else [args.variant]
    for name in names:
        run_variant(get_variant(name), args.source_dir, args.min_nodes)


if __name__ == '__main__':
    main()
