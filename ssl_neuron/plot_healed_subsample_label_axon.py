import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from plot_healed_subsample import (
    build_neighbor_map,
    subsample_swc,
    compute_graph_distances,
)


TYPE_COLORS = {
    1: "black",
    2: "firebrick",
    3: "steelblue",
}

TYPE_LABELS = {
    1: "Soma",
    2: "Axon (farthest branch)",
    3: "Dendrite",
}


def get_soma_id(df: pd.DataFrame) -> int:
    """Return soma node id, defaulting to the first row if type 1 is absent."""
    soma_candidates = df[df["type"] == 1]
    if len(soma_candidates) > 0:
        return int(soma_candidates.iloc[0]["id"])
    return int(df.iloc[0]["id"])


def identify_farthest_branch(df: pd.DataFrame, distances: dict[int, float], soma_id: int) -> set[int]:
    """Return node ids forming the branch with the maximum geodesic distance."""
    if not distances:
        return set()

    adjacency = build_neighbor_map(df)
    leaf_nodes = [
        node
        for node, neighbors in adjacency.items()
        if node != soma_id and len(neighbors) == 1
    ]

    if leaf_nodes:
        farthest_leaf = max(leaf_nodes, key=lambda node: distances.get(node, 0.0))
    else:
        farthest_leaf = max(distances, key=distances.get)

    parent_map = df.set_index("id")["parent"].astype(int).to_dict()

    branch_nodes: set[int] = set()
    current = farthest_leaf
    while current != -1:
        branch_nodes.add(current)
        current = parent_map.get(current, -1)

    return branch_nodes


def relabel_branch(df: pd.DataFrame, branch_nodes: set[int], soma_id: int) -> pd.DataFrame:
    """Assign SWC types: soma=1, farthest branch=2, remaining branches=3."""
    labeled = df.copy()
    labeled[["id", "type", "parent"]] = labeled[["id", "type", "parent"]].astype(int)
    labeled["type"] = 3
    labeled.loc[labeled["id"] == soma_id, "type"] = 1
    if branch_nodes:
        labeled.loc[labeled["id"].isin(branch_nodes - {soma_id}), "type"] = 2
    return labeled


def _collect_segments(df: pd.DataFrame, node_type: int, soma_xyz: np.ndarray, exclusion_radius: float) -> tuple[list[float], list[float], list[float]]:
    """Build line segments for a given SWC type."""
    id_indexed = df.set_index("id")
    seg_x: list[float] = []
    seg_y: list[float] = []
    seg_z: list[float] = []

    for row in df[df["type"] == node_type].itertuples(index=False):
        parent_id = int(row.parent)
        if parent_id == -1 or parent_id not in id_indexed.index:
            continue
        parent = id_indexed.loc[parent_id]

        child_xyz = np.array([row.x, row.y, row.z], dtype=float)
        if np.linalg.norm(child_xyz - soma_xyz) < exclusion_radius:
            continue

        seg_x.extend([parent.x, row.x, None])
        seg_y.extend([parent.y, row.y, None])
        seg_z.extend([parent.z, row.z, None])

    return seg_x, seg_y, seg_z


def plot_branch_labels(
    df: pd.DataFrame,
    soma_xyz: np.ndarray,
    exclusion_radius: float = 0.0,
    marker_size: int = 2,
) -> go.Figure:
    """Create a Plotly figure with soma, axon, and dendrite color coding."""
    fig = go.Figure()

    for node_type in (2, 3):
        seg_x, seg_y, seg_z = _collect_segments(df, node_type, soma_xyz, exclusion_radius)
        if not seg_x:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=seg_x,
                y=seg_y,
                z=seg_z,
                mode="lines",
                line=dict(color=TYPE_COLORS[node_type], width=2),
                name=TYPE_LABELS[node_type],
                showlegend=True,
                marker=dict(size=marker_size, color=TYPE_COLORS[node_type]),
            )
        )

    soma_nodes = df[df["type"] == 1]
    if len(soma_nodes) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=soma_nodes["x"],
                y=soma_nodes["y"],
                z=soma_nodes["z"],
                mode="markers",
                marker=dict(size=8, color=TYPE_COLORS[1], symbol="circle"),
                name=TYPE_LABELS[1],
                showlegend=True,
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    return fig


def process_neuron(
    swc_path: str,
    max_nodes: int | None = 3000,
    exclusion_radius: float = 0.0,
    show_plot: bool = False,
    output_html: str | None = None,
    output_dir: str | None = None,
    labeled_output_dir: str | None = None,
    label_suffix: str = "_axonlabel.swc",
    verbose: bool = True,
) -> tuple[pd.DataFrame, set[int], go.Figure | None]:
    """Load an SWC, label farthest branch as axon, and render a plot."""
    if verbose:
        print(f"Processing {swc_path}")

    df = pd.read_csv(
        swc_path,
        sep=" ",
        comment="#",
        header=None,
        names=["id", "type", "x", "y", "z", "r", "parent"],
    )
    df[["id", "type", "parent"]] = df[["id", "type", "parent"]].astype(int)

    soma_id = get_soma_id(df)
    soma_xyz = df.loc[df["id"] == soma_id, ["x", "y", "z"]].iloc[0].to_numpy(dtype=float)

    if max_nodes is not None and len(df) > max_nodes:
        if verbose:
            print(f"  Subsampling to at most {max_nodes} nodes...")
        df = subsample_swc(df, max_nodes=max_nodes, soma_id=soma_id)
        soma_id = get_soma_id(df)
        soma_xyz = df.loc[df["id"] == soma_id, ["x", "y", "z"]].iloc[0].to_numpy(dtype=float)

    distances = compute_graph_distances(df, soma_id)
    branch_nodes = identify_farthest_branch(df, distances, soma_id)
    labeled_df = relabel_branch(df, branch_nodes, soma_id)

    if labeled_output_dir is not None:
        target_dir = Path(labeled_output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(swc_path).stem
        output_path = target_dir / f"{base_name}{label_suffix}"
        labeled_df[["id", "type", "x", "y", "z", "r", "parent"]].to_csv(
            output_path,
            sep=" ",
            header=False,
            index=False,
        )
        if verbose:
            print(f"  Wrote labeled SWC to {output_path}")

    fig: go.Figure | None = None
    if output_html is not None or show_plot:
        fig = plot_branch_labels(labeled_df, soma_xyz, exclusion_radius=exclusion_radius)

        if output_html is not None:
            out_dir = Path(output_dir) if output_dir else Path(swc_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            html_path = out_dir / output_html
            if verbose:
                print(f"  Writing plot to {html_path}")
            fig.write_html(html_path)

        if show_plot:
            fig.show()

    return labeled_df, branch_nodes, fig


if __name__ == "__main__":
    scale = 2
    if scale == 3:
        scaletag = "3"
    elif scale == 2:
        scaletag = "2"
    else:
        raise ValueError("Scale must be 2 or 3.")

    base_dir = "/nfs/data8/chuyu/data/20230422_160839/connectome/swc"
    swc_paths = glob.glob(f"{base_dir}/skeletons/*scale{scaletag}.swc")

    ariadne_csv = os.path.join(
        base_dir,
        "Corvus_Proofreading_external - Ariadne_seeds_RoLi1_pc_matched_axonsplit_20251017_mod_8_class_degs.csv",
    )
    if os.path.exists(ariadne_csv):
        ariadne_df = pd.read_csv(ariadne_csv)
        agglo_ids = ariadne_df["agglo_id_orig"].values.astype(int)
        swc_paths = [
            f"{base_dir}/skeletons/neuron_{agglo_id}_scale{scaletag}.swc"
            for agglo_id in agglo_ids
        ]

    html_output_root = os.path.join(base_dir, "plots_swc_axon")
    labeled_output_root = os.path.join(base_dir, "axon_labeled_swcs")

    for swc_path in swc_paths:
        healed_path = swc_path.replace(".swc", "_healed.swc")
        if not os.path.exists(healed_path):
            continue

        html_filename = os.path.splitext(os.path.basename(healed_path))[0] + ".html"

        process_neuron(
            healed_path,
            max_nodes=3000,
            exclusion_radius=1.0,
            show_plot=False,
            output_html=html_filename,
            output_dir=html_output_root,
            labeled_output_dir=labeled_output_root,
            verbose=True,
        )
