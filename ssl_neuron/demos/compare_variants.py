"""Compare the three compartment variants' GraphDINO latents.

Loads `latents_<variant>.npz` for full / dendrite / axon, restricts them to the
cells all three share, and writes:

  * `tsne_variants.png`        — t-SNE of each variant, coloured by the `full`
                                 variant's k-means clusters, so you can see how
                                 much of the whole-skeleton structure survives
                                 when only one compartment is visible
  * `variant_agreement.png`    — k-means cluster agreement between variants
                                 (adjusted Rand index) and the confusion matrices
  * `latent_similarity.png`    — how similar the neighbourhood structure is:
                                 shared k-nearest-neighbours between variants
  * `compare_variants.json`    — the numbers behind the figures

Run from the repo root::

    python -m ssl_neuron.demos.compare_variants
    python -m ssl_neuron.demos.compare_variants --k 6 --out-dir /tmp/figs
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from ssl_neuron.preprocessing.variants import VARIANTS, WORK_ROOT, get_variant

ORDER = ['full', 'dendrite', 'axon']


def load_latents(names):
    """Return `{variant: (latents, cell_ids)}`, skipping variants not yet exported."""
    out = {}
    for name in names:
        path = get_variant(name).latents_path
        if not path.exists():
            print(f'skipping {name}: {path} not found')
            continue
        data = np.load(path, allow_pickle=True)
        out[name] = (data['latents'], data['cell_ids'])
        print(f'{name:9s} {data["latents"].shape} from {data["checkpoint"]}')
    return out


def align(latents):
    """Restrict every variant to the cells they all have, in a common id order."""
    shared = None
    for _, ids in latents.values():
        shared = set(ids.tolist()) if shared is None else shared & set(ids.tolist())
    shared = np.array(sorted(shared))
    aligned = {}
    for name, (z, ids) in latents.items():
        order = {int(c): i for i, c in enumerate(ids)}
        aligned[name] = z[[order[int(c)] for c in shared]]
    return aligned, shared


def zscore(z):
    return (z - z.mean(0)) / (z.std(0) + 1e-8)


def shared_neighbours(a, b, k=10):
    """Mean overlap of each cell's k nearest neighbours between two embeddings."""
    def knn(z):
        d = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d, np.inf)
        return np.argsort(d, axis=1)[:, :k]

    na, nb = knn(a), knn(b)
    return float(np.mean([len(set(x) & set(y)) for x, y in zip(na, nb)]) / k)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--k', type=int, default=5, help='number of k-means clusters')
    parser.add_argument('--knn', type=int, default=10, help='k for the shared-neighbour metric')
    parser.add_argument('--out-dir', default=str(WORK_ROOT / 'figures'))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.metrics import adjusted_rand_score, confusion_matrix, silhouette_score

    names = [n for n in ORDER if n in VARIANTS]
    latents = load_latents(names)
    if len(latents) < 2:
        raise SystemExit('need at least two exported variants to compare')

    aligned, shared_ids = align(latents)
    names = [n for n in ORDER if n in aligned]
    print(f'\n{len(shared_ids)} cells shared by {names}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # k-means per variant, on z-scored latents.
    labels, sils = {}, {}
    for name in names:
        z = zscore(aligned[name])
        labels[name] = KMeans(args.k, n_init=10, random_state=args.seed).fit_predict(z)
        sils[name] = float(silhouette_score(z, labels[name]))
        print(f'{name:9s} silhouette (k={args.k}): {sils[name]:.3f}')

    # --- t-SNE, coloured by the reference variant's clusters -----------------
    reference = names[0]
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        emb = TSNE(2, init='pca', perplexity=30, random_state=args.seed).fit_transform(
            zscore(aligned[name]))
        ax.scatter(emb[:, 0], emb[:, 1], c=labels[reference], cmap='tab10', s=14, alpha=0.85)
        ax.set_title(f'{name}\n(colour = {reference} clusters)')
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f't-SNE of GraphDINO latents, {len(shared_ids)} shared cells')
    fig.tight_layout()
    fig.savefig(out_dir / 'tsne_variants.png', dpi=150)
    plt.close(fig)

    # --- cluster agreement ---------------------------------------------------
    pairs = list(combinations(names, 2))
    ari = {f'{a}|{b}': float(adjusted_rand_score(labels[a], labels[b])) for a, b in pairs}
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.5 * len(pairs), 4))
    axes = np.atleast_1d(axes)
    for ax, (a, b) in zip(axes, pairs):
        cm = confusion_matrix(labels[a], labels[b])
        ax.imshow(cm, cmap='Blues')
        ax.set_xlabel(f'{b} cluster'); ax.set_ylabel(f'{a} cluster')
        ax.set_title(f'ARI = {ari[f"{a}|{b}"]:.3f}')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.suptitle(f'k-means agreement between variants (k={args.k})')
    fig.tight_layout()
    fig.savefig(out_dir / 'variant_agreement.png', dpi=150)
    plt.close(fig)

    # --- neighbourhood similarity -------------------------------------------
    knn_overlap = {f'{a}|{b}': shared_neighbours(zscore(aligned[a]), zscore(aligned[b]), args.knn)
                   for a, b in pairs}
    fig, ax = plt.subplots(figsize=(4 + len(pairs), 3.5))
    keys = list(knn_overlap)
    ax.bar(keys, [knn_overlap[k] for k in keys], color='steelblue')
    ax.axhline(args.knn / len(shared_ids), color='crimson', ls='--',
               label='chance')
    ax.set_ylabel(f'shared {args.knn}-NN fraction')
    ax.set_title('Do the variants place the same neurons near each other?')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'latent_similarity.png', dpi=150)
    plt.close(fig)

    summary = {
        'n_shared_cells': int(len(shared_ids)),
        'variants': names,
        'k': args.k,
        'silhouette': sils,
        'adjusted_rand_index': ari,
        f'shared_{args.knn}nn_fraction': knn_overlap,
        'chance_shared_nn_fraction': float(args.knn / len(shared_ids)),
    }
    with open(out_dir / 'compare_variants.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n' + json.dumps(summary, indent=2))
    print(f'\nfigures -> {out_dir}')


if __name__ == '__main__':
    main()
