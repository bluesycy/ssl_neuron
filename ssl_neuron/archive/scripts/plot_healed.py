import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import networkx as nx
import numpy as np
import json
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import networkx as nx
import os

def compute_graph_distances(df, soma_id):
    """Compute geodesic (graph) distances from soma using NetworkX."""
    # Build graph
    G = nx.Graph()
    
    # Add edges
    for _, row in df.iterrows():
        if row['parent'] != -1:
            parent_rows = df.loc[df['id'] == row['parent']]
            if len(parent_rows) > 0:
                parent = parent_rows.iloc[0]
                # Edge weight is Euclidean distance between nodes
                dist = np.sqrt((row['x'] - parent['x'])**2 + 
                                (row['y'] - parent['y'])**2 + 
                                (row['z'] - parent['z'])**2)
                G.add_edge(row['id'], row['parent'], weight=dist)
    
    # Compute shortest path distances from soma
    if soma_id not in G.nodes():
        # If soma not in graph, return zeros
        return {node: 0 for node in G.nodes()}
    
    distances = nx.single_source_dijkstra_path_length(G, soma_id, weight='weight')
    return distances


def plot_swc_3d_plotly(
    swc_paths,
    mesh_paths=None,
    marker_size=1,
    color_mode='neuron',  # 'neuron', 'geodist', or 'euclidean'
    scale_factors=(4, 4, 2),
    anisotropy=(80, 80, 100),
    somata=[(0, 0, 0)],
    exclusion_radius=3500,
    cmap='viridis',
    xyz_scaling=(.8, .8, 1),
    subsample_factor=1,
    rescale=False,
    reorder_axes=False,
    show_plot=True,
    output_html=None,
    output_dir=None,
):
    """
    Plot one or multiple SWC files in 3D using plotly.
    
    Parameters:
    -----------
    swc_paths : str or list of str
        Path(s) to SWC file(s)
    marker_size : float
        Size of scatter markers
    color_mode : str
        'neuron': cycle through colors per neuron (default)
        'geodist': color by graph/geodesic distance from soma
        'euclidean': color by Euclidean distance from soma
    scale_factors : tuple
        Scaling factors for coordinates
    anisotropy : tuple
        Anisotropy correction factors
    somata : list
        list of Soma coordinates (x, y, z)
    exclusion_radius : float
        Radius for filtering scatter points (points > radius are shown)
    cmap : str
        Colormap name for lines (used when color_mode is 'geodist' or 'euclidean')
    xyz_scaling : tuple
        Aspect ratio for x, y, z axes
    """
    
    # Convert single path to list
    if not isinstance(swc_paths, list):
        swc_paths = [swc_paths]
    if not isinstance(somata, list):
        somata = [somata]

    if mesh_paths is not None:
        if not isinstance(mesh_paths, list):
            mesh_paths = [mesh_paths]
    
    fig = go.Figure()
    
    # Define colors for different neurons
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 
              'lightpink', 'lightcyan', 'lavender', 'peachpuff']
    skeleton_colors = ['darkblue', 'darkred', 'darkgreen', 'orange',
                      'purple', 'darkcyan', 'indigo', 'chocolate']
    

    
    # Process each neuron
    for neuron_idx, swc_path in enumerate(swc_paths):
        
        soma = somata[neuron_idx]
        print(f"Processing neuron {neuron_idx + 1}, soma: {soma}")
        print(f"Loading SWC from: {swc_path}")
        
        # Load SWC file
        df = pd.read_csv(swc_path, sep=' ', comment='#', header=None,
                         names=['id', 'type', 'x', 'y', 'z', 'r', 'parent'])
        
        # Find soma node (typically type 1 or closest to soma coordinate)
        soma_candidates = df[df['type'] == 1]
        if len(soma_candidates) > 0:
            soma_id = soma_candidates.iloc[0]['id']
        else:
            # Find closest node to soma coordinate
            distances = np.sqrt((df['x'] - soma[0])**2 + 
                              (df['y'] - soma[1])**2 + 
                              (df['z'] - soma[2])**2)
            soma_id = df.iloc[np.argmin(distances)]['id']
        
        # Compute distances based on color mode
        if color_mode == 'geodist':
            graph_distances = compute_graph_distances(df, soma_id)
            df['distance'] = df['id'].map(graph_distances).fillna(0)
        else:  # euclidean or neuron mode
            df['distance'] = np.sqrt((df['x'] - soma[0])**2 + 
                                    (df['y'] - soma[1])**2 + 
                                    (df['z'] - soma[2])**2)
        
        # Build line segments - CRITICAL: must flatten with None separators
        seg_x, seg_y, seg_z = [], [], []
        mid_distances = []
        
        for rno, row in df.iterrows():
            if rno % subsample_factor != 0:
                continue
            
            if row['parent'] != -1:
                parent_rows = df.loc[df['id'] == row['parent']]
                if len(parent_rows) == 0:
                    continue
                parent = parent_rows.iloc[0]
                
                # Skip lines inside exclusion radius (using Euclidean for filtering)
                euclid_dist = np.sqrt((row['x'] - soma[0])**2 + 
                                     (row['y'] - soma[1])**2 + 
                                     (row['z'] - soma[2])**2)
                if euclid_dist < exclusion_radius:
                    continue
                
                # Color by midpoint distance
                mid_dist = 0.5 * (row['distance'] + parent['distance'])
                
                # Append segment with None separator for discontinuous lines
                seg_x.extend([parent['x'], row['x'], None])
                seg_y.extend([parent['y'], row['y'], None])
                seg_z.extend([parent['z'], row['z'], None])
                
                mid_distances.extend([mid_dist, mid_dist, mid_dist])
        
        # Add lines if any exist
        if len(seg_x) > 0:
            if color_mode == 'neuron':
                # Use fixed color per neuron
                color_idx = neuron_idx % len(skeleton_colors)
                parent_line = go.Scatter3d(
                    x=seg_x, 
                    y=seg_y, 
                    z=seg_z, 
                    mode='lines+markers',
                    line=dict(
                        color=skeleton_colors[color_idx],
                        width=2
                    ), 
                    marker=dict(
                        size=marker_size,
                        color=skeleton_colors[color_idx]
                    ),
                    opacity=0.8, 
                    name=f'neuron_{neuron_idx+1}',
                    showlegend=True
                )
            else:
                # Use colormap based on distance
                max_dist = df['distance'].max()
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
                        cmax=np.percentile(mid_distances, 90)
                    ), 
                    marker=dict(
                        size=marker_size,
                        color=mid_distances,
                        colorscale=cmap.capitalize(),
                        cmin=0,
                        cmax=np.percentile(mid_distances, 90)
                    ),
                    opacity=0.8, 
                    name=f'neuron_{neuron_idx+1}',
                    showlegend=True
                )
            
            fig.add_trace(parent_line)
    
    # Add mesh if provided
    if mesh_paths is not None:
        for mesh_idx, mesh_path in enumerate(mesh_paths):
            with open(mesh_path, 'r') as f:
                mesh_json = json.load(f)
            
            verts, faces = np.array(mesh_json['vertices']), np.array(mesh_json['faces'])
            x_m, y_m, z_m = verts.T
            i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
            
            # Set mesh color based on color_mode
            if color_mode == 'neuron':
                # Use corresponding neuron color (lighter version)
                color_idx = mesh_idx % len(colors)
                mesh_color = colors[color_idx]
            else:
                # Use neutral gray for distance-based coloring
                mesh_color = 'lightgrey'
            
            mesh = go.Mesh3d(
                x=x_m, 
                y=y_m, 
                z=z_m, 
                i=i, 
                j=j, 
                k=k, 
                color=mesh_color, 
                opacity=.25, 
                name=f'morphology_{mesh_idx+1}'
            )
            fig.add_trace(mesh)
    
    # Add soma spheres
    for soma in somata:
        r = 3000
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)
        u, v = np.meshgrid(u, v)
        
        # Parametric equations
        x = soma[0] + r * np.cos(u) * np.sin(v)
        y = soma[1] + r * np.sin(u) * np.sin(v)
        z = soma[2] + r * np.cos(v)
        
        # Constant color value for the whole surface
        surface_color = np.zeros_like(x)
        
        surf = go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=surface_color,     
            colorscale=[[0, "rgba(0,0,0,0.35)"], [1, "rgba(0,0,0,0.35)"]],
            lighting=dict(
                ambient=0.8,
                diffuse=0.8,
                specular=0.1,
                roughness=0.9,
                fresnel=0.2
            ),
            showscale=False               
        )
        
        fig.add_trace(surf)
    
    # Apply custom axis scaling
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data',
            aspectratio=dict(x=xyz_scaling[0], y=xyz_scaling[1], z=xyz_scaling[2])
        ),
        margin=dict(l=0, r=0, b=0, t=0)
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
        print(f"Writing figure to {target_html}...")
        fig.write_html(target_html)

    if show_plot:
        fig.show()
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
    _somata = np.array([ariadne_df[ariadne_df.agglo_id_orig == agglo_id][['x', 'y', 'z']].values for agglo_id in _agglo_ids]).squeeze()
    
    i = 0
        
    skeleton_path = _swc_paths[i]
    output_path = _swc_paths[i].replace('.swc', '_healed.swc')
    agglo_id = _agglo_ids[i]
    soma = _somata[i]


    mesh_path = f'{base_dir}/mesh/{agglo_id}.json'

    html_output_dir = os.path.join(base_dir, 'plots_original')
    html_filename = os.path.splitext(os.path.basename(output_path))[0] + '.html'

    plot_swc_3d_plotly(
        [output_path],
        mesh_paths=[mesh_path],
        somata=[soma],
        exclusion_radius=1,
        color_mode='geodist',
        show_plot=False,
        output_html=html_filename,
        output_dir=html_output_dir,
    )
