import os
import glob
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from utils import subsample_graph


def build_neighbor_map(df):
    """Build undirected adjacency from parent pointers."""
    adjacency = defaultdict(set)
    for row in df.itertuples(index=False):
        node_id = int(row.id)
        parent_id = int(row.parent)
        if parent_id != -1:
            adjacency[parent_id].add(node_id)
            adjacency[node_id].add(parent_id)
        else:
            adjacency[node_id]
    return adjacency


def prune_axon_nodes(df, axon_type=2, drop_types=(0,)):
    """Remove axon subtrees and `drop_types` nodes from an SWC.

    A neuron's SWC is a tree rooted at the soma, so anything downstream of
    an axon node is also considered axonal and is dropped. Nodes whose type
    is in `drop_types` (by default only the SWC 'undefined' slot 0) are also
    removed. Surviving nodes keep their original parent when it survived; if
    their parent was deleted, they are re-parented to the nearest surviving
    ancestor so the result is still a single tree rooted at the soma.

    Note: if the SWC parent pointers form a geometric spanning tree rather
    than a compartment-aware tree, legitimate dendrites can end up as
    descendants of axon nodes and get dropped here too. Check the
    pre/post node counts — if too much is lost, reconsider the input SWC's
    parent-pointer semantics.
    """
    df = df.copy()
    df[['id', 'type', 'parent']] = df[['id', 'type', 'parent']].astype(int)

    type_map = dict(zip(df['id'], df['type']))
    parent_map = dict(zip(df['id'], df['parent']))
    drop_types = set(drop_types)

    # Walk each node's ancestry; delete if self or any ancestor is axon, or if
    # self type is in drop_types. Memoize the verdict per node.
    verdict = {}  # id -> True if should be deleted

    def should_delete(node):
        if node in verdict:
            return verdict[node]
        path = []
        cur = node
        while cur != -1 and cur not in verdict:
            path.append(cur)
            t = type_map.get(cur)
            if t == axon_type or t in drop_types:
                for p in path:
                    verdict[p] = True
                return True
            cur = parent_map.get(cur, -1)

        inherited = verdict.get(cur, False) if cur != -1 else False
        for p in path:
            verdict[p] = inherited
        return inherited

    for node_id in df['id']:
        should_delete(int(node_id))

    keep_mask = df['id'].map(lambda i: not verdict.get(int(i), False))
    pruned = df[keep_mask].copy()

    # Re-parent any survivor whose parent was deleted to the nearest surviving
    # ancestor (in practice always the soma, since axon/drop subtrees were
    # fully removed — but guard anyway in case of disconnected inputs).
    kept_ids = set(pruned['id'].astype(int).tolist())

    def first_surviving_ancestor(node):
        p = parent_map.get(node, -1)
        while p != -1 and p not in kept_ids:
            p = parent_map.get(p, -1)
        return p

    pruned['parent'] = pruned['id'].map(lambda i: (
        int(parent_map[int(i)]) if int(parent_map[int(i)]) in kept_ids or parent_map[int(i)] == -1
        else first_surviving_ancestor(int(i))
    ))
    return pruned.reset_index(drop=True)


def subsample_swc(df, max_nodes=5000, soma_id=None):
    """Contract an SWC morphology down to at most max_nodes nodes while keeping the soma."""
    if max_nodes is None or len(df) <= max_nodes:
        return df.copy()

    subsampled = df.copy()
    subsampled[['id', 'type', 'parent']] = subsampled[['id', 'type', 'parent']].astype(int)

    if soma_id is None:
        soma_candidates = subsampled[subsampled['type'] == 1]
        if len(soma_candidates) > 0:
            soma_id = int(soma_candidates.iloc[0]['id'])
        else:
            soma_id = int(subsampled.iloc[0]['id'])
    else:
        soma_id = int(soma_id)

    adjacency = build_neighbor_map(subsampled)
    neighbors = {int(node): set(map(int, neigh)) for node, neigh in adjacency.items()}
    neighbors, kept_nodes = subsample_graph(
        neighbors=neighbors,
        not_deleted=set(neighbors.keys()),
        keep_nodes=max_nodes,
        protected=[soma_id]
    )

    parent_map = {soma_id: -1}
    stack = [soma_id]
    visited = {soma_id}
    while stack:
        node = stack.pop()
        for neighbor in neighbors.get(node, set()):
            if neighbor in visited:
                continue
            parent_map[neighbor] = node
            visited.add(neighbor)
            stack.append(neighbor)

    pruned_df = subsampled[subsampled['id'].isin(visited)].copy()
    pruned_df['parent'] = pruned_df['id'].map(parent_map).fillna(-1).astype(int)
    pruned_df = pruned_df.sort_values('id').reset_index(drop=True)
    return pruned_df


def compute_graph_distances(df, soma_id):
    """Compute geodesic distances on the contracted morphology."""
    adjacency = build_neighbor_map(df)
    coords = {int(row.id): np.array([row.x, row.y, row.z], dtype=float)
              for row in df.itertuples(index=False)}

    soma_id = int(soma_id)
    if soma_id not in adjacency:
        return {}

    distances = {soma_id: 0.0}
    stack = [soma_id]
    while stack:
        node = stack.pop()
        base_xyz = coords[node]
        base_dist = distances[node]
        for neighbor in adjacency[node]:
            if neighbor in distances:
                continue
            edge_len = float(np.linalg.norm(coords[neighbor] - base_xyz))
            distances[neighbor] = base_dist + edge_len
            stack.append(neighbor)
    return distances


def plot_swc_3d_plotly(
    swc_paths,
    mesh_paths=None,
    marker_size=1,
    color_mode='neuron',
    exclusion_radius=3500,
    cmap='viridis',
    xyz_scaling=(0.8, 0.8, 1.0),
    subsample_factor=1,
    max_nodes=None,
    trace_labels=None,
    verbose=True,
    draw_soma_markers=True,
    soma_marker_size=6,
    soma_marker_color='black',
    show_plot=True,
    output_html=None,
    output_dir=None,
    save_subsampled=False,
    subsample_output_dir=None,
    subsample_suffix='_subsampled.swc',
    prune_axon=False,
    axon_type=2,
):
    if not isinstance(swc_paths, list):
        swc_paths = [swc_paths]

    if mesh_paths is not None and not isinstance(mesh_paths, list):
        mesh_paths = [mesh_paths]

    if trace_labels is not None and not isinstance(trace_labels, (list, tuple)):
        trace_labels = [trace_labels]

    fig = go.Figure()
    computed_somata = []

    skeleton_colors = ['darkblue', 'darkred', 'darkgreen', 'orange',
                       'purple', 'darkcyan', 'indigo', 'chocolate']

    # color_mode='type': one trace per compartment type, named color per type.
    TYPE_COLORS = {
        1: ('soma', 'black'),
        2: ('axon', 'crimson'),
        3: ('dendrite', 'royalblue'),
        7: ('synapse', 'orange'),
        0: ('unknown', 'gray'),
    }

    for neuron_idx, swc_path in enumerate(swc_paths):

        if verbose:
            print(f"Processing neuron {neuron_idx + 1}/{len(swc_paths)}: {swc_path}")
            print("  Loading SWC data...")

        df = pd.read_csv(
            swc_path,
            sep=' ',
            comment='#',
            header=None,
            names=['id', 'type', 'x', 'y', 'z', 'r', 'parent'],
        )
        df[['id', 'type', 'parent']] = df[['id', 'type', 'parent']].astype(int)

        if prune_axon:
            n_before = len(df)
            df = prune_axon_nodes(df, axon_type=axon_type)
            if verbose:
                print(f"  Pruned axon nodes: {n_before} -> {len(df)} nodes.")

        if verbose:
            print(f"  Loaded {len(df)} nodes. Determining soma and subsampling...")

        soma = df[['x', 'y', 'z']].iloc[0].to_numpy(dtype=float)
        computed_somata.append(soma)

        soma_candidates = df[df['type'] == 1]
        if len(soma_candidates) > 0:
            soma_id = int(soma_candidates.iloc[0]['id'])
        else:
            distances = np.sqrt((df['x'] - soma[0])**2 +
                                (df['y'] - soma[1])**2 +
                                (df['z'] - soma[2])**2)
            soma_id = int(df.iloc[int(np.argmin(distances))]['id'])

        if max_nodes is not None and len(df) > max_nodes:
            if verbose:
                print(f"  Contracting morphology to <= {max_nodes} nodes...")
            df = subsample_swc(df, max_nodes=max_nodes, soma_id=soma_id)
            if verbose:
                print(f"  Subsampled to {len(df)} nodes.")

            if save_subsampled:
                target_dir = subsample_output_dir
                if target_dir is None:
                    target_dir = os.path.join(os.path.dirname(swc_path), 'subsampled')
                os.makedirs(target_dir, exist_ok=True)

                base_name = os.path.splitext(os.path.basename(swc_path))[0]
                suffix = subsample_suffix or '_subsampled.swc'
                if '.' not in suffix:
                    suffix = suffix + '.swc'
                subsampled_path = os.path.join(target_dir, base_name + suffix)
                df[['id', 'type', 'parent']] = df[['id', 'type', 'parent']].astype(int)
                df[['id', 'type', 'x', 'y', 'z', 'r', 'parent']].to_csv(
                    subsampled_path,
                    sep=' ',
                    header=False,
                    index=False,
                )
                if verbose:
                    print(f"  Saved subsampled SWC to {subsampled_path}.")

        if verbose and color_mode == 'geodist':
            print("  Computing geodesic distances...")

        if color_mode == 'geodist':
            graph_distances = compute_graph_distances(df, soma_id)
            df['distance'] = df['id'].map(graph_distances).fillna(0.0)
        else:
            df['distance'] = np.sqrt((df['x'] - soma[0])**2 +
                                     (df['y'] - soma[1])**2 +
                                     (df['z'] - soma[2])**2)

        if verbose:
            print("  Building plot segments...")

        seg_x, seg_y, seg_z = [], [], []
        mid_distances = []
        seg_types = []  # one entry per segment (child node's type)

        if trace_labels is not None and neuron_idx < len(trace_labels):
            trace_name = str(trace_labels[neuron_idx])
        else:
            trace_name = os.path.splitext(os.path.basename(swc_path))[0]

        for rno, row in df.iterrows():
            if subsample_factor > 1 and (rno % subsample_factor) != 0:
                continue

            if row['parent'] == -1:
                continue

            parent_rows = df.loc[df['id'] == row['parent']]
            if len(parent_rows) == 0:
                continue
            parent = parent_rows.iloc[0]

            euclid_dist = np.sqrt((row['x'] - soma[0])**2 +
                                  (row['y'] - soma[1])**2 +
                                  (row['z'] - soma[2])**2)
            if euclid_dist < exclusion_radius:
                continue

            mid_dist = 0.5 * (row['distance'] + parent['distance'])

            seg_x.extend([parent['x'], row['x'], None])
            seg_y.extend([parent['y'], row['y'], None])
            seg_z.extend([parent['z'], row['z'], None])
            mid_distances.extend([mid_dist, mid_dist, mid_dist])
            seg_types.append(int(row['type']))

        if len(seg_x) == 0:
            if verbose:
                print("  No segments survived filtering; skipping trace.")
            continue

        if verbose:
            print(f"  Adding trace '{trace_name}' with {len(seg_x)//3} segments.")

        if color_mode == 'type':
            # Emit one trace per compartment type so the legend is self-describing.
            by_type = defaultdict(lambda: {'x': [], 'y': [], 'z': []})
            for seg_idx, t in enumerate(seg_types):
                base = seg_idx * 3
                by_type[t]['x'].extend(seg_x[base:base + 3])
                by_type[t]['y'].extend(seg_y[base:base + 3])
                by_type[t]['z'].extend(seg_z[base:base + 3])

            for t in sorted(by_type):
                label, color = TYPE_COLORS.get(t, (f'type_{t}', 'gray'))
                legend_name = f'{trace_name} · {label}' if len(swc_paths) > 1 else label
                fig.add_trace(go.Scatter3d(
                    x=by_type[t]['x'],
                    y=by_type[t]['y'],
                    z=by_type[t]['z'],
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=marker_size, color=color),
                    opacity=0.85,
                    name=legend_name,
                    showlegend=True,
                ))
            continue  # traces already added; skip the fig.add_trace(parent_line) below

        if color_mode == 'neuron':
            color_idx = neuron_idx % len(skeleton_colors)
            parent_line = go.Scatter3d(
                x=seg_x,
                y=seg_y,
                z=seg_z,
                mode='lines+markers',
                line=dict(color=skeleton_colors[color_idx], width=2),
                marker=dict(size=marker_size, color=skeleton_colors[color_idx]),
                opacity=0.8,
                name=trace_name,
                showlegend=True,
            )
        else:
            parent_line = go.Scatter3d(
                x=seg_x,
                y=seg_y,
                z=seg_z,
                mode='lines+markers',
                line=dict(
                    color=mid_distances,
                    colorscale=cmap.capitalize(),
                    width=2,
                    cmin=0,
                    cmax=np.percentile(mid_distances, 90),
                ),
                marker=dict(
                    size=marker_size,
                    color=mid_distances,
                    colorscale=cmap.capitalize(),
                    cmin=0,
                    cmax=np.percentile(mid_distances, 90),
                ),
                opacity=0.8,
                name=trace_name,
                showlegend=True,
            )

        fig.add_trace(parent_line)

    if mesh_paths is not None:
        if verbose:
            print("Adding mesh overlays...")
        for mesh_idx, mesh_path in enumerate(mesh_paths):
            with open(mesh_path, 'r') as f:
                mesh_json = json.load(f)
            verts = np.array(mesh_json['vertices'])
            faces = np.array(mesh_json['faces'])
            mesh = go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightgrey',
                opacity=0.25,
                name=f'mesh_{mesh_idx}',
            )
            fig.add_trace(mesh)

    if draw_soma_markers and computed_somata:
        if verbose:
            print("Adding soma markers...")
        soma_x = [soma[0] for soma in computed_somata]
        soma_y = [soma[1] for soma in computed_somata]
        soma_z = [soma[2] for soma in computed_somata]
        soma_trace = go.Scatter3d(
            x=soma_x,
            y=soma_y,
            z=soma_z,
            mode='markers',
            marker=dict(
                size=soma_marker_size,
                color=soma_marker_color,
                symbol='circle',
                opacity=0.9,
            ),
            name='soma',
            showlegend=False,
        )
        fig.add_trace(soma_trace)
    elif verbose:
        print("Skipping soma markers.")

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data',
            aspectratio=dict(x=xyz_scaling[0], y=xyz_scaling[1], z=xyz_scaling[2]),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    if output_html is not None:
        target_html = output_html
        target_dir = None
        if output_dir is not None:
            if not os.path.isabs(target_html):
                target_html = os.path.join(output_dir, target_html)
                target_dir = output_dir
            else:
                target_dir = os.path.dirname(target_html)
        else:
            target_dir = os.path.dirname(target_html)

        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        if verbose:
            print(f"Writing figure to {target_html}...")
        fig.write_html(target_html)

    if show_plot:
        if verbose:
            print("Rendering interactive figure...")
        fig.show()
    elif verbose:
        print("Skipping interactive render.")

    if verbose:
        print("Plot ready.")
    return fig


if __name__ == "__main__":
    # SWC type column convention (from collaborator):
    #   1 = soma, 2 = axon, 3 = dendrite, 7 = synapse (glia slot, repurposed)
    # Soma is identified directly from type == 1 in subsample_swc / plot_swc_3d_plotly,
    # so no external CSV lookup is needed.
    base_dir = '/nfs/data8/chuyu/data/20230422_160839/connectome/skeletons/healed_syns_swc/swc'

    # One file in this dir is a gzipped tar bundle that re-packs the same 467
    # neurons already present as plain SWCs — skip it.
    skip_names = {'neuron_136593859_scale2_healed_syns.swc'}
    swc_paths = sorted(
        p for p in glob.glob(os.path.join(base_dir, '*.swc'))
        if os.path.basename(p) not in skip_names
    )

    # Toggle: also produce a parallel set of outputs with axon subtrees removed
    # *before* subsampling, so the 3000-node budget goes entirely to dendrites.
    prune_axon = False

    if prune_axon:
        html_output_root = os.path.join(base_dir, 'plots_html_no_axon')
        subsampled_output_root = os.path.join(base_dir, 'subsampled_swcs_no_axon')
    else:
        html_output_root = os.path.join(base_dir, 'plots_html')
        subsampled_output_root = os.path.join(base_dir, 'subsampled_swcs')

    for swc_path in swc_paths:
        html_filename = os.path.splitext(os.path.basename(swc_path))[0] + '.html'

        plot_swc_3d_plotly(
            [swc_path],
            exclusion_radius=1,
            color_mode='type',
            max_nodes=3000,
            subsample_factor=1,
            verbose=True,
            soma_marker_size=8,
            show_plot=False,
            output_html=html_filename,
            output_dir=html_output_root,
            save_subsampled=True,
            subsample_output_dir=subsampled_output_root,
            prune_axon=prune_axon,
            axon_type=2,
        )

