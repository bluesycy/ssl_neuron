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

    for neuron_idx, swc_path in enumerate(swc_paths):

        if verbose:
            print(f"Processing neuron {neuron_idx + 1}/{len(swc_paths)}: {swc_path}")
            print("  Loading SWC data...")

        df = pd.read_csv(
            swc_path,
            sep=' ',
            comment='#',
            header=None,
            names=['id', 'type', 'x', 'y', 'z', 'r', 'parent']
        )
        df[['id', 'type', 'parent']] = df[['id', 'type', 'parent']].astype(int)

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

        if len(seg_x) == 0:
            if verbose:
                print("  No segments survived filtering; skipping trace.")
            continue

        if verbose:
            print(f"  Adding trace '{trace_name}' with {len(seg_x)//3} segments.")

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
    scale = 2
    if scale == 3:
        scale_factors = (8, 8, 4)
        anisotropy = (10*8, 10*8, 25*4)
        scaletag = '3'

    elif scale == 2:
        scale_factors = (4, 4, 2)
        anisotropy = (10*4, 10*4, 25*2)
        scaletag = '2'
    else:
        raise ValueError("Scale must be 2 or 3.")
    
    # change accordingly
    base_dir = f'/nfs/data8/chuyu/data/20230422_160839/connectome/swc'

    swc_paths = glob.glob(f'{base_dir}/skeletons/*scale{scaletag}.swc') 
    ariadne_csv = os.path.join(base_dir, 'Corvus_Proofreading_external - Ariadne_seeds_RoLi1_pc_matched_axonsplit_20251017_mod_8_class_degs.csv')

    ariadne_df = pd.read_csv(ariadne_csv)

    _agglo_ids = ariadne_df['agglo_id_orig'].values.astype(int)
    _swc_paths = [f'{base_dir}/skeletons/neuron_{agglo_id}_scale{scaletag}.swc' for agglo_id in _agglo_ids]
    
    html_output_root = os.path.join(base_dir, 'plots_swc_mesh')
    subsampled_output_root = os.path.join(base_dir, 'subsampled_swcs')

    for agglo_id, swc_path in zip(_agglo_ids, _swc_paths):
        healed_path = swc_path.replace('.swc', '_healed.swc')
        mesh_path = f'{base_dir}/mesh/{agglo_id}.json'

        html_filename = os.path.splitext(os.path.basename(healed_path))[0] + '.html'

        plot_swc_3d_plotly(
            [healed_path],
            # mesh_paths=[mesh_path],
            exclusion_radius=1,
            color_mode='geodist',
            max_nodes=3000,
            subsample_factor=1,
            verbose=True,
            soma_marker_size=8,
            show_plot=False,
            output_html=html_filename,
            output_dir=html_output_root,
            save_subsampled=True,
            subsample_output_dir=subsampled_output_root,
        )

