"""Reading, writing and graph-building for the connectome SWC skeletons.

SWC type column convention used by this dataset (from the collaborator, and
*not* the stock SWC spec):

    0 = undefined   1 = soma   2 = axon   3 = dendrite   7 = synapse

Coordinates are in EM nanometres. Row 0 of every file is the soma.
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd

SWC_COLUMNS = ['id', 'type', 'x', 'y', 'z', 'r', 'parent']

# SWC type codes.
UNDEFINED = 0
SOMA = 1
AXON = 2
DENDRITE = 3
SYNAPSE = 7

# Compartments a node can be assigned to (see compartments.resolve_compartments).
COMPARTMENTS = (SOMA, AXON, DENDRITE)

# One file in the source directory is a gzipped tar bundle that re-packs the
# same neurons already present as plain SWCs — never read it as an SWC.
BAD_SWC_FILES = {'neuron_136593859_scale2_healed_syns.swc'}


def read_swc(path):
    """Read an SWC file into a DataFrame with integer id/type/parent columns."""
    df = pd.read_csv(path, sep=r'\s+', comment='#', header=None, names=SWC_COLUMNS)
    df[['id', 'type', 'parent']] = df[['id', 'type', 'parent']].astype(int)
    return df


def write_swc(df, path):
    """Write a morphology DataFrame back out in SWC column order."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    out = df.copy()
    out[['id', 'type', 'parent']] = out[['id', 'type', 'parent']].astype(int)
    out[SWC_COLUMNS].to_csv(path, sep=' ', header=False, index=False)


def cell_id_from_path(path):
    """`neuron_<cell_id>_scale2_healed_syns[...].swc` -> int cell_id.

    cell_id is the neuron's segment_id (see the repo's paths.json).
    """
    return int(os.path.basename(path).split('_')[1])


def list_swc_files(directory, pattern='*.swc'):
    """Sorted SWC paths in `directory`, minus the known-bad tarball."""
    import glob

    return sorted(
        p for p in glob.glob(os.path.join(directory, pattern))
        if os.path.basename(p) not in BAD_SWC_FILES
    )


def build_neighbors(df):
    """Undirected adjacency `{node_id: set(neighbour ids)}` from parent pointers.

    Every node in `df` gets a key, including isolated ones.
    """
    neighbors = defaultdict(set)
    ids = df['id'].to_numpy(dtype=int)
    parents = df['parent'].to_numpy(dtype=int)
    present = set(ids.tolist())

    for node_id in ids:
        neighbors[int(node_id)]  # touch, so isolated nodes still get a key
    for node_id, parent_id in zip(ids, parents):
        if parent_id != -1 and int(parent_id) in present:
            neighbors[int(parent_id)].add(int(node_id))
            neighbors[int(node_id)].add(int(parent_id))
    return dict(neighbors)


def soma_id(df):
    """Id of the soma node: the first `type == 1` row, else the root, else row 0."""
    soma_rows = df.index[df['type'] == SOMA]
    if len(soma_rows) > 0:
        return int(df.loc[soma_rows[0], 'id'])
    root_rows = df.index[df['parent'] == -1]
    if len(root_rows) > 0:
        return int(df.loc[root_rows[0], 'id'])
    return int(df.iloc[0]['id'])


def tree_from_neighbors(neighbors, root):
    """Rebuild parent pointers by BFS from `root` over an undirected graph.

    Returns `{node: parent}` with `root -> -1`, covering only the component
    that contains `root`.
    """
    parent = {int(root): -1}
    stack = [int(root)]
    while stack:
        node = stack.pop()
        for neighbor in neighbors.get(node, ()):  # noqa: B007
            if neighbor in parent:
                continue
            parent[neighbor] = node
            stack.append(neighbor)
    return parent


def renumber(df, correction=True):
    """Renumber ids to 0..N-1 in row order and remap parents accordingly.

    With `correction=True` the root's type is forced to soma, matching the
    original notebook's `read_swc_file(correction=True)`.
    """
    df = df.copy()
    if correction:
        df.loc[df['parent'] == -1, 'type'] = SOMA
    id_map = {old: new for new, old in enumerate(df['id'].astype(int))}
    df['id'] = df['id'].map(id_map)
    df['parent'] = df['parent'].map(lambda p: id_map.get(int(p), -1))
    df = df.sort_values('id').reset_index(drop=True)
    df['id'] = df.index
    return df


def euclidean_lengths(df):
    """Length of each node's edge to its parent (0 for the root), in input units."""
    xyz = df[['x', 'y', 'z']].to_numpy(dtype=float)
    pos = {int(i): xyz[k] for k, i in enumerate(df['id'].to_numpy(dtype=int))}
    out = np.zeros(len(df))
    for k, (node_id, parent_id) in enumerate(
        zip(df['id'].to_numpy(dtype=int), df['parent'].to_numpy(dtype=int))
    ):
        if parent_id != -1 and parent_id in pos:
            out[k] = float(np.linalg.norm(pos[int(node_id)] - pos[int(parent_id)]))
    return out
