"""SWC -> GraphDINO dataset preprocessing.

The pipeline has two stages, each with a CLI:

    subsample.py      full SWC -> compartment mask -> <=max_nodes SWC
    build_dataset.py  subsampled SWC -> features.npy / neighbors.pkl / id splits

See `variants.py` for the compartment variants (full / dendrite / axon) and
`ssl_neuron/preprocessing/README.md` for the layout of the generated data.
"""

from ssl_neuron.preprocessing.variants import VARIANTS, Variant  # noqa: F401
