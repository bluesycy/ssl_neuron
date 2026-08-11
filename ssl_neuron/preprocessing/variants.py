"""The three compartment variants we embed, and where their data lives.

Every variant is trained *with* node-type information (feat_dim=7), so the
models differ only in which part of the morphology they see.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Optional

from ssl_neuron.preprocessing.swc_io import AXON, DENDRITE, SOMA

# Root of the connectome skeleton data (see the repo's paths.json).
# /nfs/roli8/data/chuyu/... is the same directory via a different mount.
SKELETON_ROOT = Path(os.environ.get(
    'SSL_NEURON_SKELETON_ROOT',
    '/nfs/data8/chuyu/20230422_160839/connectome/skeletons/healed_syns_swc',
))

#: Source SWCs, one file per neuron, full unsubsampled morphology.
SOURCE_SWC_DIR = SKELETON_ROOT / 'swc'

#: Everything this pipeline generates lives under here.
WORK_ROOT = SKELETON_ROOT / 'graphdino'

#: Node budget after subsampling (per variant).
MAX_NODES = 3000

#: Cells with fewer than this many nodes after masking are dropped — GraphDINO
#: needs at least `data.n_nodes` (200) nodes to build a view.
MIN_NODES = 200


@dataclass(frozen=True)
class Variant:
    name: str
    keep: Optional[FrozenSet[int]]
    description: str

    @property
    def swc_dir(self) -> Path:
        """Masked + subsampled SWCs for this variant."""
        return WORK_ROOT / 'swc_subsampled' / self.name

    @property
    def dataset_dir(self) -> Path:
        """GraphDINO dataset root: skeletons/ plus the id splits."""
        return WORK_ROOT / 'datasets' / self.name

    @property
    def ckpt_dir(self) -> Path:
        return WORK_ROOT / 'ckpts' / self.name

    @property
    def latents_path(self) -> Path:
        return WORK_ROOT / 'embeddings' / f'latents_{self.name}.npz'


VARIANTS = {
    v.name: v for v in [
        Variant(
            name='full',
            keep=None,
            description='whole skeleton (soma + dendrite + axon), with type info',
        ),
        Variant(
            name='dendrite',
            keep=frozenset({SOMA, DENDRITE}),
            description='soma + dendrites only, with type info',
        ),
        Variant(
            name='axon',
            keep=frozenset({SOMA, AXON}),
            description='soma + axon only, with type info',
        ),
    ]
}


def get_variant(name: str) -> Variant:
    try:
        return VARIANTS[name]
    except KeyError:
        raise SystemExit(
            f'unknown variant {name!r}; choose one of {sorted(VARIANTS)}'
        ) from None
