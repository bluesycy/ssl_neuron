"""Tests for compartment resolution and masking.

Run with:  pytest ssl_neuron/preprocessing/test_compartments.py
"""

import numpy as np
import pandas as pd

from ssl_neuron.preprocessing.build_dataset import one_hot_slots
from ssl_neuron.preprocessing.compartments import mask_compartments, resolve_compartments
from ssl_neuron.preprocessing.swc_io import AXON, DENDRITE, SOMA, SYNAPSE, UNDEFINED


def toy_tree():
    """A tree exercising every interesting case, including the real quirk that
    an axon can emerge from a dendrite and carry further dendrite below it."""
    rows = [
        (0, SOMA, 0, 0, 0, 1, -1),
        (1, DENDRITE, 1, 0, 0, 1, 0),
        (2, SYNAPSE, 2, 0, 0, 1, 1),    # synapse on a dendrite
        (3, AXON, 3, 0, 0, 1, 1),       # axon emerging from a dendrite
        (4, SYNAPSE, 4, 0, 0, 1, 3),    # synapse on the axon
        (5, DENDRITE, 5, 0, 0, 1, 3),   # dendrite hanging below the axon
        (6, UNDEFINED, 6, 0, 0, 1, 0),  # unlabelled straight off the soma
        (7, UNDEFINED, 7, 0, 0, 1, 6),
        (8, UNDEFINED, 8, 0, 0, 1, 5),  # unlabelled below a dendrite
        (9, SYNAPSE, 9, 0, 0, 1, 0),    # synapse on the soma
    ]
    return pd.DataFrame(rows, columns=['id', 'type', 'x', 'y', 'z', 'r', 'parent'])


def test_resolve_compartments():
    comp = resolve_compartments(toy_tree())
    assert list(comp) == [
        SOMA,       # 0 soma
        DENDRITE,   # 1 own type
        DENDRITE,   # 2 synapse inherits the dendrite it sits on
        AXON,       # 3 own type, even though its parent is a dendrite
        AXON,       # 4 synapse inherits the axon
        DENDRITE,   # 5 own type wins over its axon parent
        UNDEFINED,  # 6 the soma must not propagate its label
        UNDEFINED,  # 7 inherits 6's non-assignment
        DENDRITE,   # 8 inherits the dendrite above it
        UNDEFINED,  # 9 synapse on the soma is assignable to neither neurite
    ]


def test_mask_keeps_single_tree():
    df = toy_tree()
    for keep in ({SOMA, DENDRITE}, {SOMA, AXON}):
        masked = mask_compartments(df, keep)
        assert (masked['parent'] == -1).sum() == 1, 'must stay a single tree'
        kept = set(masked['id'])
        assert 0 in kept, 'soma must always survive'
        assert set(masked.loc[masked['id'] != 0, 'parent']).issubset(kept)


def test_mask_reparents_to_nearest_surviving_ancestor():
    # Node 5 is a dendrite whose parent (3) is an axon; in the dendrite variant
    # it must re-attach to node 1, not jump all the way to the soma.
    masked = mask_compartments(toy_tree(), {SOMA, DENDRITE}).set_index('id')
    assert int(masked.loc[5, 'parent']) == 1
    assert set(masked.index) == {0, 1, 2, 5, 8}


def test_mask_drops_unassignable_nodes():
    masked = mask_compartments(toy_tree(), {SOMA, AXON})
    assert set(masked['id']) == {0, 3, 4}, 'unlabelled + soma synapses are dropped'


def test_full_variant_keeps_everything():
    df = toy_tree()
    assert len(mask_compartments(df, None)) == len(df)


def test_one_hot_slots():
    slots = one_hot_slots(toy_tree())
    # [soma, axon, dendrite, synapse] slot indices; -1 means an all-zero vector.
    assert list(slots) == [0, 2, 3, 1, 3, 2, -1, -1, 2, 3]


def test_unlabelled_cell_collapses_to_soma():
    """A reconstruction with no compartment labels at all must not survive as a
    whole neuron in both compartment variants."""
    rows = [(0, SOMA, 0, 0, 0, 1, -1)]
    rows += [(i, UNDEFINED, i, 0, 0, 1, i - 1) for i in range(1, 50)]
    df = pd.DataFrame(rows, columns=['id', 'type', 'x', 'y', 'z', 'r', 'parent'])
    for keep in ({SOMA, DENDRITE}, {SOMA, AXON}):
        assert len(mask_compartments(df, keep)) == 1
    assert len(mask_compartments(df, None)) == 50


def test_features_have_no_nans():
    df = toy_tree()
    slots = one_hot_slots(df)
    assert not np.any(np.isnan(slots.astype(float)))
