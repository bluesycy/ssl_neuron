"""Stage 1: full SWC -> compartment mask -> subsampled SWC.

Masking happens *before* subsampling so each variant spends its whole node
budget on the compartment it cares about. Doing it the other way round (masking
the already-subsampled 3000-node files) starves the axon variant: 194 of 466
cells would end up under 1000 nodes, versus 23 under 3000 when masking first.

Usage (from the repo root, with the `ssl` conda env active):

    python -m ssl_neuron.preprocessing.subsample --variant dendrite
    python -m ssl_neuron.preprocessing.subsample --variant axon
    python -m ssl_neuron.preprocessing.subsample --variant all --jobs 16

The subsampled files keep their original SWC node ids; renumbering to 0..N-1
happens in `build_dataset.py`.
"""

import argparse
import multiprocessing as mp
import sys
from functools import partial
from pathlib import Path

import numpy as np

from ssl_neuron.preprocessing import compartments as comp_mod
from ssl_neuron.preprocessing import swc_io
from ssl_neuron.preprocessing.variants import (
    MAX_NODES,
    MIN_NODES,
    SOURCE_SWC_DIR,
    VARIANTS,
    get_variant,
)
from ssl_neuron.utils import subsample_graph


def contract_to(df, max_nodes, root_id):
    """Contract the morphology to at most `max_nodes` nodes, keeping the soma.

    Uses the same edge-contraction routine as the training-time augmentation
    (`utils.subsample_graph`), then rebuilds parent pointers by BFS from the
    soma so the result is still a valid SWC tree.
    """
    if max_nodes is None or len(df) <= max_nodes:
        return df.copy().reset_index(drop=True)

    neighbors = swc_io.build_neighbors(df)
    neighbors, _ = subsample_graph(
        neighbors=neighbors,
        not_deleted=set(neighbors.keys()),
        keep_nodes=max_nodes,
        protected=[int(root_id)],
    )

    parent = swc_io.tree_from_neighbors(neighbors, root_id)
    out = df[df['id'].isin(parent.keys())].copy()
    out['parent'] = out['id'].map(lambda i: parent[int(i)])
    return out.sort_values('id').reset_index(drop=True)


def process_one(path, keep, max_nodes=MAX_NODES, min_nodes=MIN_NODES, out_dir=None):
    """Mask + subsample a single SWC. Returns a stats dict for the report."""
    path = Path(path)
    cell_id = swc_io.cell_id_from_path(path)
    try:
        df = swc_io.read_swc(path)
        n_source = len(df)
        root_id = swc_io.soma_id(df)

        masked = comp_mod.mask_compartments(df, keep)
        masked = comp_mod.largest_component(masked, root_id=root_id)
        n_masked = len(masked)

        if n_masked < min_nodes:
            return dict(cell_id=cell_id, n_source=n_source, n_masked=n_masked,
                        n_out=0, status='too_small')

        out = contract_to(masked, max_nodes, root_id)
        swc_io.write_swc(out, str(Path(out_dir) / f'{path.stem}.swc'))
        return dict(cell_id=cell_id, n_source=n_source, n_masked=n_masked,
                    n_out=len(out), status='ok')
    except Exception as exc:  # keep one bad file from killing the whole run
        return dict(cell_id=cell_id, n_source=0, n_masked=0, n_out=0,
                    status=f'error: {exc}')


def run_variant(variant, source_dir, max_nodes, min_nodes, jobs, limit=None):
    paths = swc_io.list_swc_files(source_dir)
    if limit:
        paths = paths[:limit]

    out_dir = variant.swc_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[{variant.name}] {len(paths)} source SWCs -> {out_dir}', flush=True)

    worker = partial(process_one, keep=variant.keep, max_nodes=max_nodes,
                     min_nodes=min_nodes, out_dir=str(out_dir))

    if jobs > 1:
        with mp.Pool(jobs) as pool:
            results = []
            for i, res in enumerate(pool.imap_unordered(worker, paths, chunksize=1), 1):
                results.append(res)
                if i % 25 == 0 or i == len(paths):
                    print(f'[{variant.name}] {i}/{len(paths)}', flush=True)
    else:
        results = [worker(p) for p in paths]

    ok = [r for r in results if r['status'] == 'ok']
    small = [r for r in results if r['status'] == 'too_small']
    errors = [r for r in results if r['status'].startswith('error')]

    sizes = np.array([r['n_out'] for r in ok]) if ok else np.array([0])
    print(f'[{variant.name}] wrote {len(ok)} files '
          f'(nodes: min={sizes.min()} median={int(np.median(sizes))} max={sizes.max()}), '
          f'skipped {len(small)} below {min_nodes} nodes, {len(errors)} errors')
    for r in small:
        print(f'    skipped {r["cell_id"]}: {r["n_masked"]} nodes after mask')
    for r in errors:
        print(f'    ERROR   {r["cell_id"]}: {r["status"]}', file=sys.stderr)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', default='all',
                        help=f'one of {sorted(VARIANTS)} or "all"')
    parser.add_argument('--source-dir', default=str(SOURCE_SWC_DIR),
                        help='directory of full source SWCs')
    parser.add_argument('--max-nodes', type=int, default=MAX_NODES)
    parser.add_argument('--min-nodes', type=int, default=MIN_NODES)
    parser.add_argument('--jobs', type=int, default=max(1, mp.cpu_count() // 2))
    parser.add_argument('--limit', type=int, default=None,
                        help='only process the first N cells (smoke test)')
    args = parser.parse_args()

    names = sorted(VARIANTS) if args.variant == 'all' else [args.variant]
    for name in names:
        run_variant(get_variant(name), args.source_dir, args.max_nodes,
                    args.min_nodes, args.jobs, args.limit)


if __name__ == '__main__':
    main()
