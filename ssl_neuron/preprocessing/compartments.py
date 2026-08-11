"""Assign every SWC node to a compartment and mask the tree down to a subset.

Why this exists
---------------
These skeletons' parent pointers are *not* compartment-aware: 18.5% of dendrite
nodes sit downstream of an axon node and 92% of axon nodes sit downstream of a
dendrite node. So deleting whole subtrees (the old `prune_axon_nodes`) throws
away ~18% of genuine dendrite. Masking by node type and re-parenting survivors
is the correct operation, and in practice it barely fragments the tree: masking
the full SWCs leaves a median of 1-2 orphan components per cell (max 36 over the
466 cells), which re-parenting stitches straight back onto the surviving tree.

Compartment assignment
----------------------
A node's compartment is its own type when that is soma/axon/dendrite. Synapse
(7) and undefined (0) nodes inherit from their nearest *axon or dendrite*
ancestor, so post-synaptic sites follow the dendrite and pre-synaptic boutons
follow the axon.

The soma deliberately does **not** propagate its label downward. A node that
reaches the soma without passing through a labelled neurite is genuinely
unassigned (UNDEFINED) and is dropped from the compartment-specific variants.
Without this, the 12 cells that are >50% type-0 — reconstructions that carry no
compartment labels at all — would survive whole in *both* the dendrite and the
axon variant and contaminate the comparison.

The node's *type* column is never rewritten, so the saved one-hot still
distinguishes synapse nodes inside every variant.
"""

import numpy as np

from ssl_neuron.preprocessing.swc_io import (
    AXON,
    DENDRITE,
    SOMA,
    UNDEFINED,
    build_neighbors,
    tree_from_neighbors,
)

#: Only these types pass their compartment on to unlabelled descendants.
PROPAGATING = (AXON, DENDRITE)


def resolve_compartments(df):
    """Return an int array, aligned to `df` rows, of each node's compartment.

    Values are SOMA / AXON / DENDRITE, or UNDEFINED when the node has no
    axon-or-dendrite ancestor to inherit from.
    """
    ids = df['id'].to_numpy(dtype=int)
    types = df['type'].to_numpy(dtype=int)
    parents = df['parent'].to_numpy(dtype=int)
    row_of = {int(i): k for k, i in enumerate(ids)}

    # inherit[row] = compartment an unlabelled node hanging off `row` receives.
    # It is the row's own type when that propagates (axon/dendrite), otherwise
    # whatever the row itself inherits. The soma and the root yield UNDEFINED,
    # which is what stops the soma's label from leaking into its children.
    inherit = {}

    def inherited_at(row):
        path = []
        while True:
            if row in inherit:
                value = inherit[row]
                break
            if types[row] in PROPAGATING:
                value = int(types[row])
                break
            if types[row] == SOMA:
                value = UNDEFINED
                break
            parent_id = parents[row]
            if parent_id == -1 or int(parent_id) not in row_of:
                value = UNDEFINED
                break
            path.append(row)
            row = row_of[int(parent_id)]
        inherit[row] = value
        for r in path:
            inherit[r] = value
        return value

    comp = np.empty(len(ids), dtype=int)
    for row in range(len(ids)):
        if types[row] == SOMA:
            comp[row] = SOMA
        elif types[row] in PROPAGATING:
            comp[row] = int(types[row])
        else:
            parent_id = parents[row]
            comp[row] = (
                inherited_at(row_of[int(parent_id)])
                if parent_id != -1 and int(parent_id) in row_of
                else UNDEFINED
            )
    return comp


def mask_compartments(df, keep, protect_soma=True):
    """Keep only nodes whose compartment is in `keep`, re-parenting survivors.

    Survivors whose parent was dropped are re-attached to their nearest
    surviving ancestor, so the result stays a single tree rooted at the soma.
    Any node left with no surviving ancestor is attached to the soma.

    Args:
        df: morphology DataFrame (see swc_io.read_swc).
        keep: iterable of compartments to keep, e.g. `{SOMA, DENDRITE}`.
            `None` keeps everything (the `full` variant).
        protect_soma: always keep the soma row even if its compartment is
            not in `keep`.

    Returns:
        A new DataFrame with original ids preserved and `parent` rewritten.
    """
    if keep is None:
        return df.copy().reset_index(drop=True)

    keep = set(int(k) for k in keep)
    ids = df['id'].to_numpy(dtype=int)
    parents = df['parent'].to_numpy(dtype=int)
    row_of = {int(i): k for k, i in enumerate(ids)}

    comp = resolve_compartments(df)
    keep_mask = np.isin(comp, list(keep))

    root_rows = np.where(parents == -1)[0]
    root_row = int(root_rows[0]) if len(root_rows) else 0
    if protect_soma:
        keep_mask[root_row] = True

    if not keep_mask.any():
        return df.iloc[[root_row]].assign(parent=-1).reset_index(drop=True)

    root_id = int(ids[root_row])

    def nearest_surviving_ancestor(row):
        parent_id = parents[row]
        while parent_id != -1 and int(parent_id) in row_of:
            prow = row_of[int(parent_id)]
            if keep_mask[prow]:
                return int(ids[prow])
            parent_id = parents[prow]
        return -1

    new_parent = np.full(len(ids), -1, dtype=int)
    for row in np.where(keep_mask)[0]:
        if row == root_row:
            continue
        parent_id = parents[row]
        if parent_id != -1 and int(parent_id) in row_of and keep_mask[row_of[int(parent_id)]]:
            new_parent[row] = int(parent_id)
        else:
            ancestor = nearest_surviving_ancestor(row)
            new_parent[row] = ancestor if ancestor != -1 else root_id

    out = df[keep_mask].copy()
    out['parent'] = new_parent[keep_mask]
    out.loc[out['id'] == root_id, 'parent'] = -1
    return out.reset_index(drop=True)


def largest_component(df, root_id=None):
    """Restrict `df` to the connected component containing the soma.

    `mask_compartments` already returns a single tree, but source files can
    contain stray disconnected nodes; this drops them so downstream code never
    has to call the O(N^2) `connect_graph` repair.
    """
    if root_id is None:
        root_rows = df.index[df['parent'] == -1]
        root_id = int(df.loc[root_rows[0], 'id']) if len(root_rows) else int(df.iloc[0]['id'])

    neighbors = build_neighbors(df)
    reachable = tree_from_neighbors(neighbors, root_id)
    if len(reachable) == len(df):
        return df.reset_index(drop=True)

    out = df[df['id'].isin(reachable.keys())].copy()
    out['parent'] = out['id'].map(lambda i: reachable[int(i)])
    return out.reset_index(drop=True)


def compartment_counts(df):
    """`{compartment: n_nodes}` — handy for logging and sanity checks."""
    comp = resolve_compartments(df)
    return {
        'soma': int((comp == SOMA).sum()),
        'axon': int((comp == AXON).sum()),
        'dendrite': int((comp == DENDRITE).sum()),
        'unassigned': int((comp == UNDEFINED).sum()),
    }
