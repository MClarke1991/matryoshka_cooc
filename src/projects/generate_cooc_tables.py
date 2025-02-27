import logging
import os
import pickle
import time
from os.path import join as pj
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from matryoshka_cooc.sae import GlobalBatchTopKMatryoshkaSAE


def set_device() -> str:
    """Set device based on availability"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def setup_logging(log_file: str) -> None:
    """Setup logging to file and console"""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def create_directories(base_path: str) -> dict[str, str]:
    """Create required directories for saving results"""
    paths = {
        "base": base_path,
        "histograms": pj(base_path, "histograms"),
        "dataframes": pj(base_path, "dataframes"),
        "thresholded_matrices": pj(base_path, "thresholded_matrices"),
        "subgraph_objects": pj(base_path, "subgraph_objects"),
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        
    return paths


def load_cooccurrence_data(results_dir: str, sae_name: str, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Load co-occurrence and feature activation data"""
    threshold_str = str(threshold).replace(".", "_")
    
    cooccurrence_file = pj(results_dir, f"{sae_name}_cooccurrence_{threshold_str}.npz")
    activations_file = pj(results_dir, f"{sae_name}_activations_{threshold_str}.npz")
    
    # Load co-occurrence matrix
    with np.load(cooccurrence_file) as data:
        array_key = list(data.keys())[0]  # Get the first key
        cooccurrence = data[array_key]
    
    # Load feature activations
    with np.load(activations_file) as data:
        array_key = list(data.keys())[0]
        feature_activations = data[array_key]
    
    return cooccurrence, feature_activations


def remove_self_loops_inplace(matrix: np.ndarray) -> None:
    """Remove self-loops from a matrix by setting diagonal to 0 (in-place)"""
    np.fill_diagonal(matrix, 0)


def find_threshold(
    matrix: np.ndarray,
    min_size: int = 150,
    max_size: int = 200,
    tolerance: float = 1e-3,
) -> tuple[float, int]:
    """
    Find an edge weight threshold that results in subgraphs of desired size range
    
    Args:
        matrix: Adjacency matrix
        min_size: Minimum desired size of largest connected component
        max_size: Maximum desired size of largest connected component
        tolerance: Tolerance for binary search
        
    Returns:
        Tuple of (threshold value, size of largest component)
    """
    import scipy.sparse as sparse
    from scipy.sparse.csgraph import connected_components
    
    sparse_matrix = sparse.csr_matrix(matrix)
    low, high = 0.0, 1.0
    best_threshold = None
    best_size = None

    logging.info(f"Finding threshold with size range {min_size}-{max_size}...")
    
    def largest_component_size(sparse_matrix: sparse.csr_matrix, threshold: float) -> int:
        """Calculate size of largest connected component"""
        binary_matrix = sparse_matrix >= threshold
        n_components, labels = connected_components(
            binary_matrix, directed=False, connection="weak"
        )
        return int(np.max(np.bincount(labels)))

    while high - low > tolerance:
        mid = (low + high) / 2
        size = largest_component_size(sparse_matrix, mid)
        
        logging.info(f"Threshold: {mid:.4f}, largest component size: {size}")

        if min_size <= size <= max_size:
            return mid, size  # Early return if size is within desired range

        if size < min_size:
            high = mid
        else:  # size > max_size
            low = mid

        # Update best found so far
        if best_size is None or abs(size - (min_size + max_size) / 2) < abs(
            best_size - (min_size + max_size) / 2
        ):
            best_threshold = mid
            best_size = size

    if best_threshold is None or best_size is None:
        raise ValueError("No threshold found within the specified range.")

    return best_threshold, best_size


def remove_low_weight_edges(matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Remove edges with weights below threshold
    
    Args:
        matrix: Input adjacency matrix
        threshold: Edge weight threshold
        
    Returns:
        Thresholded matrix
    """
    thresholded = matrix.copy()
    thresholded[thresholded < threshold] = 0
    return thresholded


def plot_subgraph_size_density(
    subgraphs: list[set],
    save_dir: str,
    filename: str,
    min_size: int,
    max_size: int,
) -> None:
    """Plot distribution of subgraph sizes"""
    import matplotlib.pyplot as plt
    
    # Extract subgraph sizes
    subgraph_sizes = [len(subgraph) for subgraph in subgraphs]

    # Create a density plot
    plt.figure(figsize=(8, 6))
    plt.hist(subgraph_sizes, bins=30, density=True, color="skyblue")

    # Add dotted vertical lines at min and max sizes
    plt.axvline(min_size, color="r", linestyle="--", label=f"Min Size: {min_size}")
    plt.axvline(max_size, color="g", linestyle="--", label=f"Max Size: {max_size}")

    plt.xlabel("Subgraph Size")
    plt.ylabel("Density")
    plt.title(
        f"Density Plot of Subgraph Sizes\nMin Size: {min_size}, Max Size: {max_size}"
    )
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.yscale("log")
    plt.legend()

    # Save the plot
    plt.savefig(pj(save_dir, f"{filename}.png"), dpi=300)
    plt.close()


def create_node_info_dataframe(
    subgraphs: list[nx.Graph],
    feature_activations: np.ndarray,
    sae_name: str,
    threshold: float,
) -> pd.DataFrame:
    """
    Create a dataframe with node information for each subgraph
    
    Args:
        subgraphs: List of subgraphs
        feature_activations: Array of feature activations
        sae_name: Name of the SAE model
        threshold: Activation threshold used
        
    Returns:
        DataFrame with node information
    """
    import pandas as pd
    
    node_info_data = []
    
    for i, subgraph in enumerate(tqdm(subgraphs, desc="Creating node info dataframe")):
        subgraph_size = len(subgraph)
        for node in subgraph:
            node_info = {
                "node_id": node,
                "sae_name": sae_name,
                "activity_threshold": threshold,
                "subgraph_id": i,
                "subgraph_size": subgraph_size,
                "feature_activations": feature_activations[node],
            }
            node_info_data.append(node_info)
    
    # Add isolated nodes (not in any subgraph)
    all_nodes = set(range(len(feature_activations)))
    subgraph_nodes = set().union(*subgraphs) if subgraphs else set()
    isolated_nodes = all_nodes - subgraph_nodes
    
    for node in isolated_nodes:
        node_info = {
            "node_id": node,
            "sae_name": sae_name,
            "activity_threshold": threshold,
            "subgraph_id": "Not in subgraph",
            "subgraph_size": 1,
            "feature_activations": feature_activations[node],
        }
        node_info_data.append(node_info)
    
    return pd.DataFrame(node_info_data)


def process_cooccurrence_data(
    cooccurrence: np.ndarray,
    feature_activations: np.ndarray,
    sae_name: str,
    threshold: float,
    paths: dict[str, str],
    min_subgraph_size: int = 150,
    max_subgraph_size: int = 200,
) -> tuple[float, np.ndarray, list[nx.Graph], pd.DataFrame]:
    """
    Process co-occurrence data to generate graphs and subgraphs
    
    Args:
        cooccurrence: Co-occurrence matrix
        feature_activations: Feature activations
        sae_name: Name of the SAE model
        threshold: Activation threshold used
        paths: Dictionary of output paths
        min_subgraph_size: Minimum desired subgraph size
        max_subgraph_size: Maximum desired subgraph size
        
    Returns:
        Tuple of (edge_threshold, thresholded_matrix, subgraphs, node_info_df)
    """
    # Step 1: Remove self-loops
    remove_self_loops_inplace(cooccurrence)
    
    # Step 2: Find edge threshold
    edge_threshold, largest_size = find_threshold(
        cooccurrence, min_subgraph_size, max_subgraph_size
    )
    logging.info(f"Selected edge threshold: {edge_threshold:.4f}, largest component size: {largest_size}")
    
    # Step 3: Create thresholded matrix
    thresholded_matrix = remove_low_weight_edges(cooccurrence, edge_threshold)
    
    # Step 4: Create graph and get subgraphs
    graph = nx.from_numpy_array(thresholded_matrix)
    subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    
    # Step 5: Plot subgraph size distribution
    connected_components = list(nx.connected_components(graph))
    plot_subgraph_size_density(
        connected_components,
        paths["histograms"],
        f"{sae_name}_subgraph_density_{str(threshold).replace('.', '_')}",
        min_subgraph_size,
        max_subgraph_size,
    )
    
    # Step 6: Create node info dataframe
    node_info_df = create_node_info_dataframe(
        connected_components, feature_activations, sae_name, threshold
    )
    
    # Save node info dataframe
    safe_threshold = str(threshold).replace(".", "_")
    node_info_df.to_csv(
        pj(paths["dataframes"], f"{sae_name}_node_info_{safe_threshold}.csv"), index=False
    )
    
    # Save thresholded matrix
    np.savez_compressed(
        pj(paths["thresholded_matrices"], f"{sae_name}_thresholded_matrix_{safe_threshold}.npz"),
        thresholded_matrix,
    )
    
    # Save sparse matrix
    import scipy.sparse as sparse
    sparse_matrix = sparse.csr_matrix(thresholded_matrix)
    sparse.save_npz(
        pj(paths["thresholded_matrices"], f"{sae_name}_sparse_thresholded_matrix_{safe_threshold}.npz"),
        sparse_matrix,
    )
    
    # Save subgraphs
    # for i, subgraph in enumerate(subgraphs):
    #     subgraph_dir = pj(paths["subgraph_objects"], f"{sae_name}_activation_{safe_threshold}")
    #     os.makedirs(subgraph_dir, exist_ok=True)
    #     with open(pj(subgraph_dir, f"subgraph_{i}.pkl"), "wb") as f:
    #         pickle.dump(subgraph, f)
    
    return edge_threshold, thresholded_matrix, subgraphs, node_info_df


def save_edge_thresholds(edge_thresholds: dict, save_path: str) -> None:
    """Save edge thresholds to a CSV file"""
    import pandas as pd
    
    thresholds_data = []
    for sae_name, thresholds in edge_thresholds.items():
        for activation_threshold, edge_threshold in thresholds.items():
            thresholds_data.append({
                "sae_name": sae_name,
                "activation_threshold": activation_threshold,
                "edge_threshold": edge_threshold
            })
    
    thresholds_df = pd.DataFrame(thresholds_data)
    thresholds_df.to_csv(pj(save_path, "edge_thresholds.csv"), index=False)
    logging.info("Edge thresholds saved.")


def process_sae_cooccurrence(
    sae_name: str,
    input_dir: str,
    output_dir: str,
    activation_thresholds: list[float],
    min_subgraph_size: int = 150,
    max_subgraph_size: int = 200,
) -> dict[float, float]:
    """
    Process co-occurrence data for a single SAE
    
    Args:
        sae_name: Name of the SAE model
        input_dir: Directory with co-occurrence data
        output_dir: Directory for output files
        activation_thresholds: List of activation thresholds to process
        min_subgraph_size: Minimum desired subgraph size
        max_subgraph_size: Maximum desired subgraph size
        
    Returns:
        Dictionary mapping activation thresholds to edge thresholds
    """
    logging.info(f"Processing {sae_name} SAE with {len(activation_thresholds)} thresholds")
    
    # Create output directories
    output_path = pj(output_dir, sae_name)
    paths = create_directories(output_path)
    
    # Process each threshold
    edge_thresholds = {}
    
    for threshold in activation_thresholds:
        logging.info(f"Processing threshold {threshold} for {sae_name}")
        
        # Load co-occurrence data
        cooccurrence, feature_activations = load_cooccurrence_data(
            input_dir, sae_name, threshold
        )
        
        # Process data
        edge_threshold, _, _, _ = process_cooccurrence_data(
            cooccurrence,
            feature_activations,
            sae_name,
            threshold,
            paths,
            min_subgraph_size,
            max_subgraph_size,
        )
        
        edge_thresholds[threshold] = edge_threshold
    
    return edge_thresholds


def main() -> None:
    """Main function to process co-occurrence data and generate graphs"""
    # Configuration
    input_dir = "cooccurrence_results_layer_8"  # Directory containing co-occurrence matrices
    output_dir = "graph_results_layer_8"  # Directory for output
    activation_thresholds = [0.0, 1.5]  # Thresholds to process
    min_subgraph_size = 150
    max_subgraph_size = 200
    
    # SAE names to process
    sae_names = ["matryoshka", "resjb"]
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(pj(output_dir, "graph_generation.log"))
    
    logging.info("Starting graph generation")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Activation thresholds: {activation_thresholds}")
    logging.info(f"Subgraph size range: {min_subgraph_size}-{max_subgraph_size}")
    
    start_time = time.time()
    
    # Process each SAE
    all_edge_thresholds = {}
    
    for sae_name in sae_names:
        if not os.path.exists(pj(input_dir, f"{sae_name}_cooccurrence_{str(activation_thresholds[0]).replace('.', '_')}.npz")):
            logging.warning(f"Co-occurrence data for {sae_name} not found, skipping")
            continue
            
        edge_thresholds = process_sae_cooccurrence(
            sae_name=sae_name,
            input_dir=input_dir,
            output_dir=output_dir,
            activation_thresholds=activation_thresholds,
            min_subgraph_size=min_subgraph_size,
            max_subgraph_size=max_subgraph_size,
        )
        
        all_edge_thresholds[sae_name] = edge_thresholds
    
    # Save edge thresholds
    save_edge_thresholds(all_edge_thresholds, output_dir)
    
    # Done
    execution_time = time.time() - start_time
    logging.info(f"Graph generation complete in {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()