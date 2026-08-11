"""Interactive 3D plots of SWC skeletons (plotly).

Reads whatever SWCs you point it at — source, masked or subsampled — and writes
one self-contained HTML per neuron. Colour by compartment type, by neuron, or by
distance from the soma.

Usage:

    # QC the generated dendrite-variant skeletons
    python -m ssl_neuron.preprocessing.plot_swc --variant dendrite --limit 20

    # or an arbitrary directory
    python -m ssl_neuron.preprocessing.plot_swc --swc-dir /path/to/swc --out-dir /path/to/html
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from ssl_neuron.preprocessing import swc_io
from ssl_neuron.preprocessing.subsample import contract_to
from ssl_neuron.preprocessing.variants import VARIANTS, get_variant

# Compartment colours, keyed by raw SWC type.
TYPE_COLORS = {
    swc_io.SOMA: ('soma', 'black'),
    swc_io.AXON: ('axon', 'crimson'),
    swc_io.DENDRITE: ('dendrite', 'royalblue'),
    swc_io.SYNAPSE: ('synapse', 'orange'),
    swc_io.UNDEFINED: ('undefined', 'gray'),
}

SKELETON_COLORS = ['darkblue', 'darkred', 'darkgreen', 'orange',
                   'purple', 'darkcyan', 'indigo', 'chocolate']


def compute_graph_distances(df, root_id):
    """Geodesic distance from the soma to every node, along the skeleton."""
    neighbors = swc_io.build_neighbors(df)
    xyz = df[['x', 'y', 'z']].to_numpy(dtype=float)
    pos = {int(i): xyz[k] for k, i in enumerate(df['id'].to_numpy(dtype=int))}

    root_id = int(root_id)
    if root_id not in neighbors:
        return {}

    distances = {root_id: 0.0}
    stack = [root_id]
    while stack:
        node = stack.pop()
        for neighbor in neighbors[node]:
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node] + float(
                np.linalg.norm(pos[neighbor] - pos[node]))
            stack.append(neighbor)
    return distances


def plot_swc_3d(
    swc_paths,
    mesh_paths=None,
    marker_size=1,
    color_mode='type',
    exclusion_radius=0,
    cmap='viridis',
    xyz_scaling=(0.8, 0.8, 1.0),
    max_nodes=None,
    trace_labels=None,
    verbose=True,
    draw_soma_markers=True,
    soma_marker_size=6,
    soma_marker_color='black',
    show_plot=False,
    output_html=None,
):
    """Plot one or more SWC skeletons in 3D.

    Args:
        swc_paths: path or list of paths to SWC files.
        color_mode: 'type' (one trace per compartment), 'neuron' (one colour
            per file) or 'geodist' / anything else (colour by distance).
        exclusion_radius: hide segments closer than this to the soma, in the
            SWC's own units (nm here). 0 keeps everything.
        max_nodes: contract each morphology to at most this many nodes first.
    """
    import plotly.graph_objs as go

    if not isinstance(swc_paths, list):
        swc_paths = [swc_paths]
    if mesh_paths is not None and not isinstance(mesh_paths, list):
        mesh_paths = [mesh_paths]
    if trace_labels is not None and not isinstance(trace_labels, (list, tuple)):
        trace_labels = [trace_labels]

    fig = go.Figure()
    somata = []

    for neuron_idx, swc_path in enumerate(swc_paths):
        if verbose:
            print(f'Processing {neuron_idx + 1}/{len(swc_paths)}: {swc_path}')

        df = swc_io.read_swc(swc_path)
        root_id = swc_io.soma_id(df)
        soma_xyz = df.loc[df['id'] == root_id, ['x', 'y', 'z']].to_numpy(dtype=float)[0]
        somata.append(soma_xyz)

        if max_nodes is not None and len(df) > max_nodes:
            df = contract_to(df, max_nodes, root_id)
            if verbose:
                print(f'  contracted to {len(df)} nodes')

        if color_mode == 'geodist':
            graph_distances = compute_graph_distances(df, root_id)
            df['distance'] = df['id'].map(graph_distances).fillna(0.0)
        else:
            df['distance'] = np.linalg.norm(
                df[['x', 'y', 'z']].to_numpy(dtype=float) - soma_xyz, axis=1)

        # Build parent->child line segments, separated by None so plotly breaks the line.
        by_id = df.set_index('id')
        seg_x, seg_y, seg_z, mid_distances, seg_types = [], [], [], [], []
        for row in df.itertuples(index=False):
            if row.parent == -1 or row.parent not in by_id.index:
                continue
            if exclusion_radius and np.linalg.norm(
                    np.array([row.x, row.y, row.z]) - soma_xyz) < exclusion_radius:
                continue
            parent = by_id.loc[row.parent]
            mid = 0.5 * (row.distance + parent['distance'])
            seg_x.extend([parent['x'], row.x, None])
            seg_y.extend([parent['y'], row.y, None])
            seg_z.extend([parent['z'], row.z, None])
            mid_distances.extend([mid, mid, mid])
            seg_types.append(int(row.type))

        if not seg_x:
            if verbose:
                print('  no segments survived filtering; skipping')
            continue

        if trace_labels is not None and neuron_idx < len(trace_labels):
            trace_name = str(trace_labels[neuron_idx])
        else:
            trace_name = os.path.splitext(os.path.basename(swc_path))[0]

        if color_mode == 'type':
            by_type = defaultdict(lambda: {'x': [], 'y': [], 'z': []})
            for seg_idx, t in enumerate(seg_types):
                base = seg_idx * 3
                by_type[t]['x'].extend(seg_x[base:base + 3])
                by_type[t]['y'].extend(seg_y[base:base + 3])
                by_type[t]['z'].extend(seg_z[base:base + 3])
            for t in sorted(by_type):
                label, color = TYPE_COLORS.get(t, (f'type_{t}', 'gray'))
                name = f'{trace_name} · {label}' if len(swc_paths) > 1 else label
                fig.add_trace(go.Scatter3d(
                    x=by_type[t]['x'], y=by_type[t]['y'], z=by_type[t]['z'],
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=marker_size, color=color),
                    opacity=0.85, name=name, showlegend=True))
            continue

        if color_mode == 'neuron':
            color = SKELETON_COLORS[neuron_idx % len(SKELETON_COLORS)]
            line = dict(color=color, width=2)
            marker = dict(size=marker_size, color=color)
        else:
            cmax = float(np.percentile(mid_distances, 90))
            line = dict(color=mid_distances, colorscale=cmap.capitalize(),
                        width=2, cmin=0, cmax=cmax)
            marker = dict(size=marker_size, color=mid_distances,
                          colorscale=cmap.capitalize(), cmin=0, cmax=cmax)

        fig.add_trace(go.Scatter3d(x=seg_x, y=seg_y, z=seg_z, mode='lines+markers',
                                   line=line, marker=marker, opacity=0.8,
                                   name=trace_name, showlegend=True))

    if mesh_paths:
        for mesh_idx, mesh_path in enumerate(mesh_paths):
            with open(mesh_path) as f:
                mesh_json = json.load(f)
            verts = np.array(mesh_json['vertices'])
            faces = np.array(mesh_json['faces'])
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='lightgrey', opacity=0.25, name=f'mesh_{mesh_idx}'))

    if draw_soma_markers and somata:
        somata = np.array(somata)
        fig.add_trace(go.Scatter3d(
            x=somata[:, 0], y=somata[:, 1], z=somata[:, 2], mode='markers',
            marker=dict(size=soma_marker_size, color=soma_marker_color,
                        symbol='circle', opacity=0.9),
            name='soma', showlegend=False))

    fig.update_layout(
        scene=dict(xaxis=dict(title='X'), yaxis=dict(title='Y'), zaxis=dict(title='Z'),
                   aspectmode='data',
                   aspectratio=dict(x=xyz_scaling[0], y=xyz_scaling[1], z=xyz_scaling[2])),
        margin=dict(l=0, r=0, b=0, t=0))

    if output_html:
        os.makedirs(os.path.dirname(output_html) or '.', exist_ok=True)
        fig.write_html(output_html)
        if verbose:
            print(f'  wrote {output_html}')
    if show_plot:
        fig.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', default=None,
                        help=f'plot a variant\'s subsampled SWCs ({sorted(VARIANTS)})')
    parser.add_argument('--swc-dir', default=None, help='explicit SWC directory')
    parser.add_argument('--out-dir', default=None, help='where to write the HTML files')
    parser.add_argument('--color-mode', default='type', choices=['type', 'neuron', 'geodist'])
    parser.add_argument('--max-nodes', type=int, default=None)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    if args.variant:
        variant = get_variant(args.variant)
        swc_dir = args.swc_dir or str(variant.swc_dir)
        out_dir = args.out_dir or str(variant.swc_dir.parent.parent / 'plots_html' / variant.name)
    else:
        if not args.swc_dir:
            raise SystemExit('pass --variant or --swc-dir')
        swc_dir = args.swc_dir
        out_dir = args.out_dir or os.path.join(swc_dir, 'plots_html')

    paths = swc_io.list_swc_files(swc_dir)
    if args.limit:
        paths = paths[:args.limit]
    print(f'{len(paths)} SWCs from {swc_dir} -> {out_dir}')

    for path in paths:
        html = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + '.html')
        plot_swc_3d([path], color_mode=args.color_mode, max_nodes=args.max_nodes,
                    soma_marker_size=8, show_plot=False, output_html=html)


if __name__ == '__main__':
    main()
