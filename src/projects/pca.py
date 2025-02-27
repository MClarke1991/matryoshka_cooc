#!/usr/bin/env python3
"""
Generate PCA visualizations for a Matryoshka SAE model.

This script analyzes the structure of activations in a trained Matryoshka SAE
by performing PCA on activation patterns from specific subgraphs. The script
generates various visualizations to help understand the latent space structure.
"""

import argparse
import logging
import os
import pickle
from datetime import datetime
from os.path import join as pj
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from matryoshka_cooc.activation_store import ActivationsStore
from matryoshka_cooc.config import get_default_cfg, post_init_cfg
from matryoshka_cooc.sae import GlobalBatchTopKMatryoshkaSAE


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


def set_device() -> str:
    """Set device based on availability"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def create_directories(base_path: str) -> dict[str, str]:
    """Create required directories for saving results"""
    paths = {
        "base": base_path,
        "plots": pj(base_path, "plots"),
        "data": pj(base_path, "data"),
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        
    return paths


def load_matryoshka_sae(checkpoint_path: str) -> tuple[GlobalBatchTopKMatryoshkaSAE, dict[str, Any]]:
    """
    Load a trained matryoshka SAE model
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Tuple of SAE model and configuration
    """
    device = set_device()
    logging.info(f"Using device: {device}")

    # Configuration for GPT2-small
    cfg = get_default_cfg()
    cfg["model_name"] = "gpt2-small"
    cfg["layer"] = 0
    cfg["site"] = "resid_pre"
    cfg["act_size"] = 768
    cfg["device"] = device

    # Matryoshka-specific configurations
    cfg["sae_type"] = "global-matryoshka-topk"
    cfg["dict_size"] = 768 * 32  # Total dictionary size
    cfg["top_k"] = 32
    cfg["group_sizes"] = [768, 768, 768 * 2, 768 * 4, 768 * 8, 768 * 16]

    # Update config with derived values
    cfg = post_init_cfg(cfg)

    # Create model with the same architecture
    sae = GlobalBatchTopKMatryoshkaSAE(cfg)

    # Load trained weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    sae.load_state_dict(state_dict)
    logging.info(f"Loaded Matryoshka SAE from {checkpoint_path}")

    return sae, cfg


def load_cooccurrence_data(results_dir: str, sae_name: str, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Load co-occurrence and feature activation data"""
    threshold_str = str(threshold).replace(".", "_")
    
    cooccurrence_file = pj(results_dir, f"{sae_name}_cooccurrence_{threshold_str}.npz")
    activations_file = pj(results_dir, f"{sae_name}_activations_{threshold_str}.npz")
    
    logging.info(f"Loading co-occurrence data from {cooccurrence_file}")
    logging.info(f"Loading feature activations from {activations_file}")
    
    # Load co-occurrence matrix
    with np.load(cooccurrence_file) as data:
        array_key = list(data.keys())[0]  # Get the first key
        cooccurrence = data[array_key]
    
    # Load feature activations
    with np.load(activations_file) as data:
        array_key = list(data.keys())[0]
        feature_activations = data[array_key]
    
    return cooccurrence, feature_activations


def load_node_info(results_dir: str, sae_name: str, threshold: float) -> pd.DataFrame:
    """Load node information DataFrame"""
    threshold_str = str(threshold).replace(".", "_")
    node_info_file = pj(results_dir, "dataframes", f"{sae_name}_node_info_{threshold_str}.csv")
    
    logging.info(f"Loading node info from {node_info_file}")
    
    if not os.path.exists(node_info_file):
        raise FileNotFoundError(f"Node info file not found: {node_info_file}")
    
    return pd.read_csv(node_info_file)


def list_flatten(nested_list):
    """Flattens a nested list into a single-level list."""
    return [x for y in nested_list for x in y]


def make_token_df(tokens, model, len_prefix=5, len_suffix=5):
    """Create a DataFrame containing token information and context for each token in the input."""
    str_tokens = [model.to_str_tokens(t) for t in tokens]
    unique_token = [
        [f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens
    ]

    context, batch, pos, label = [], [], [], []
    for b in range(tokens.shape[0]):
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p - len_prefix) : p])
            suffix = "".join(
                str_tokens[b][p + 1 : min(tokens.shape[1], p + 1 + len_suffix)]
            )
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            batch.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")

    return pd.DataFrame(
        {
            "str_tokens": list_flatten(str_tokens),
            "unique_token": list_flatten(unique_token),
            "context": context,
            "batch": batch,
            "pos": pos,
            "label": label,
        }
    )


def get_special_tokens(model):
    """Get a set of special tokens for the model"""
    special_tokens = set()
    
    # Add common special tokens
    special_tokens.add(model.tokenizer.bos_token_id)
    special_tokens.add(model.tokenizer.eos_token_id)
    special_tokens.add(model.tokenizer.pad_token_id)
    
    # Filter out None values
    special_tokens = {token for token in special_tokens if token is not None}
    
    return special_tokens


def run_model_with_cache(model, tokens, sae):
    """Run the model with caching and return the cached activations for the SAE hook."""
    # For matryoshka SAE, the hook point is in the config
    hook_point = sae.config["hook_point"]
    layer = sae.config.get("layer", 0)
    
    _, cache = model.run_with_cache(
        tokens,
        stop_at_layer=layer + 1,
        names_filter=[hook_point],
    )
    return cache[hook_point]


def get_max_feature_info(feature_acts, fired_mask, feature_list_tensor, device="cpu"):
    """Calculate maximum feature information for fired tokens."""
    max_feature_values, max_feature_indices = feature_acts[fired_mask].max(dim=1)
    max_feature_in_graph = torch.tensor(
        [float(idx in feature_list_tensor) for idx in max_feature_indices],
        dtype=torch.float32,
        device=device,
    )
    max_feature_info = torch.stack(
        [max_feature_values, max_feature_indices.float(), max_feature_in_graph],
        dim=1,
    )
    return max_feature_info


def get_top_tokens_and_context(all_fired_tokens, all_token_dfs):
    """Get the top 3 most common tokens and an example context for the most common token."""
    # Validate all tokens exist in DataFrame
    token_set = set(all_token_dfs["str_tokens"])
    for token in all_fired_tokens:
        if token not in token_set:
            raise IndexError(f"Token '{token}' not found in DataFrame")

    from collections import Counter
    token_counts = Counter(all_fired_tokens)
    if token_counts:
        top_3_tokens = token_counts.most_common(3)
        most_common_token = top_3_tokens[0][0]
        example_context = all_token_dfs.loc[
            all_token_dfs["str_tokens"] == most_common_token
        ]["context"].iloc[0]
    else:
        top_3_tokens = []
        example_context = ""
    return top_3_tokens, example_context


class ProcessedExamples:
    """
    A class to store processed examples from the model.
    """
    def __init__(
        self,
        all_token_dfs,
        all_fired_tokens,
        all_reconstructions,
        all_graph_feature_acts,
        all_examples_found,
        all_max_feature_info,
        # all_feature_acts,
        top_3_tokens=None,
        example_context="",
    ):
        self.all_token_dfs = all_token_dfs
        self.all_fired_tokens = all_fired_tokens
        self.all_reconstructions = all_reconstructions
        self.all_graph_feature_acts = all_graph_feature_acts
        # self.all_feature_acts = all_feature_acts
        self.all_max_feature_info = all_max_feature_info
        self.all_examples_found = all_examples_found
        self.top_3_tokens = top_3_tokens if top_3_tokens is not None else []
        self.example_context = example_context


def process_examples(
    activation_store,
    model,
    sae,
    feature_list,
    n_batches_reconstruction,
    remove_special_tokens=False,
    device="cpu",
    max_examples=500,
):
    """
    Process examples from the activation store using the given model and SAE
    """
    examples_found = 0
    all_fired_tokens = []
    all_graph_feature_acts = []
    all_max_feature_info = []
    # all_feature_acts = []
    all_reconstructions = []
    all_token_dfs = []

    feature_list_tensor = torch.tensor(feature_list, device=sae.W_dec.device)

    pbar = tqdm(range(n_batches_reconstruction), leave=False)
    for _ in pbar:
        # Get a batch of tokens from the activation store
        tokens = activation_store.get_batch_tokens()

        # Create a DataFrame containing token information
        tokens_df = make_token_df(tokens, model)

        # Flatten the tokens tensor for easier processing
        flat_tokens = tokens.flatten()

        # Run the model and get activations
        sae_in = run_model_with_cache(model, tokens, sae)
        # Encode the activations using the SAE
        feature_acts = sae.encode(sae_in).squeeze()
        # Flatten the feature activations to 2D
        feature_acts = feature_acts.flatten(0, 1).to(device)

        # Create a mask for tokens where any feature in the feature_list is activated
        fired_mask = (feature_acts[:, feature_list]).sum(dim=-1) > 0
        fired_mask = fired_mask.to(device)

        if remove_special_tokens:
            special_tokens = get_special_tokens(model)
            # Create a mask for non-special tokens
            non_special_mask = ~torch.isin(
                flat_tokens,
                torch.tensor(list(special_tokens), device=device),
            )
            # Combine the fired_mask with the non_special_mask
            fired_mask = fired_mask & non_special_mask

        # Convert the fired tokens to string representations
        fired_tokens = model.to_str_tokens(flat_tokens[fired_mask])

        # Reconstruct the activations for the fired tokens using only the features in the feature_list
        reconstruction = (
            feature_acts[fired_mask][:, feature_list] @ sae.W_dec[feature_list]
        )

        # Get max feature info
        max_feature_info = get_max_feature_info(
            feature_acts, fired_mask, feature_list_tensor, device=device
        )

        # Append the rows of tokens_df where fired_mask is True
        # Convert fired_mask to CPU, get non-zero indices, flatten, and convert to numpy
        # Use these indices to select rows from tokens_df
        all_token_dfs.append(
            tokens_df.iloc[fired_mask.cpu().nonzero().flatten().numpy()]
        )
        # Append feature activations for fired tokens, filtered by feature_list
        all_graph_feature_acts.append(feature_acts[fired_mask][:, feature_list])

        # Append all feature activations for fired tokens
        # all_feature_acts.append(feature_acts[fired_mask]) 

        # Append maximum feature information for fired tokens
        all_max_feature_info.append(max_feature_info)

        # Append the string representations of fired tokens
        all_fired_tokens.append(fired_tokens)

        # Append reconstructions for fired tokens using selected features
        all_reconstructions.append(reconstruction)

        examples_found += len(fired_tokens)
        pbar.set_description(f"Examples found: {examples_found}")

        # Add early termination check
        if max_examples is not None and examples_found >= max_examples:
            break

    logging.info(f"Total examples found: {examples_found}")
    # Flatten the list of lists
    all_token_dfs = pd.concat(all_token_dfs) if all_token_dfs else pd.DataFrame()
    all_fired_tokens = list_flatten(all_fired_tokens)
    all_reconstructions = torch.cat(all_reconstructions) if all_reconstructions else torch.tensor([])
    all_graph_feature_acts = torch.cat(all_graph_feature_acts) if all_graph_feature_acts else torch.tensor([])
    # all_feature_acts = torch.cat(all_feature_acts) if all_feature_acts else torch.tensor([]) 
    all_max_feature_info = torch.cat(all_max_feature_info) if all_max_feature_info else torch.tensor([])

    top_3_tokens, example_context = get_top_tokens_and_context(
        all_fired_tokens, all_token_dfs
    ) if all_fired_tokens and not all_token_dfs.empty else ([], "")

    return ProcessedExamples(
        all_token_dfs=all_token_dfs,
        all_fired_tokens=all_fired_tokens,
        all_reconstructions=all_reconstructions,
        all_graph_feature_acts=all_graph_feature_acts,
        all_examples_found=examples_found,
        all_max_feature_info=all_max_feature_info,
        # all_feature_acts=all_feature_acts,
        top_3_tokens=top_3_tokens,
        example_context=example_context,
    )


def perform_pca_on_results(
    results: ProcessedExamples,
    n_components: int = 3,
    method: str = "full",
):
    """
    Perform PCA on the reconstructions from ProcessedExamples and return a DataFrame with the results.
    """
    # Get dimensions of the data
    n_samples, n_features = results.all_reconstructions.detach().cpu().numpy().shape
    max_components = min(n_samples, n_features)

    if n_components > max_components:
        import warnings
        warnings.warn(
            f"Cannot perform PCA: requested n_components ({n_components}) is greater than "
            f"max possible components ({max_components}). Returning None.",
            UserWarning,
        )
        return None, None

    # Perform PCA
    pca = PCA(n_components=n_components, svd_solver=method)
    pca_embedding = pca.fit_transform(results.all_reconstructions.cpu().numpy())

    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        pca_embedding, columns=pd.Index([f"PC{i+1}" for i in range(n_components)])
    )

    # Add tokens and context information
    pca_df["tokens"] = results.all_fired_tokens
    pca_df["context"] = results.all_token_dfs.context.values
    pca_df["point_id"] = range(len(pca_df))

    return pca_df, pca


def calculate_pca_decoder(sae, feature_list):
    """Perform PCA on the decoder weights for a list of features."""
    # Perform PCA
    pca = PCA(n_components=3)
    pca_embedding = pca.fit_transform(sae.W_dec[feature_list].cpu().numpy())
    pca_df = pd.DataFrame(
        pca_embedding, columns=pd.Index([f"PC{i+1}" for i in range(3)])
    )
    return pca, pca_df


def generate_data(
    model,
    sae,
    activation_store,
    feature_list,
    n_batches_reconstruction,
    decoder=False,
    remove_special_tokens=False,
    device="cpu",
    max_examples=5_000_000,
):
    """Generate PCA data for a set of features."""
    results = process_examples(
        activation_store,
        model,
        sae,
        feature_list,
        n_batches_reconstruction,
        remove_special_tokens,
        device=device,
        max_examples=max_examples,
    )
    
    pca_df, pca = perform_pca_on_results(results, n_components=3)
    
    if decoder:
        pca_decoder, pca_decoder_df = calculate_pca_decoder(sae, feature_list)
    else:
        pca_decoder = None
        pca_decoder_df = None

    return {
        "results": results,
        "pca_df": pca_df,
        "pca": pca,
        "pca_decoder": pca_decoder,
        "pca_decoder_df": pca_decoder_df,
    }


def save_data_to_pickle(data: dict[str, Any], file_path: str):
    """Save data to a pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    logging.info(f"Data saved to {file_path}")


def load_data_from_pickle(file_path: str) -> dict[str, Any]:
    """Load data from a pickle file."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    logging.info(f"Data loaded from {file_path}")
    return data


def plot_token_pca(pca_df, subgraph_id, title="PCA Analysis", show=True, save_path=None):
    """Create a scatter plot visualization of PCA results by token."""
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Get unique tokens for coloring
    unique_tokens = pca_df["tokens"].unique()
    
    # Use a good color scheme that will work with many tokens
    color_scale = px.colors.qualitative.Dark24
    color_map = {token: color_scale[i % len(color_scale)] for i, token in enumerate(unique_tokens)}
    
    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("PC1 vs PC2", "PC1 vs PC3", "PC2 vs PC3")
    )
    
    # Add traces for each projection
    for i, (x, y) in enumerate([("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")]):
        for token in unique_tokens:
            mask = pca_df["tokens"] == token
            fig.add_trace(
                go.Scatter(
                    x=pca_df[mask][x],
                    y=pca_df[mask][y],
                    mode="markers",
                    marker=dict(color=color_map[token]),
                    name=token,
                    text=pca_df[mask]["context"],
                    hoverinfo="text",
                    showlegend=(i == 0),  # Only show legend for the first subplot
                ),
                row=1,
                col=i+1,
            )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1800,
        title_text=f"{title} - Subgraph {subgraph_id}",
        legend_title_text="Token",
    )
    
    # Save if requested
    if save_path:
        fig.write_image(f"{save_path}/pca_plot_subgraph_{subgraph_id}_by_token.png")
        fig.write_html(f"{save_path}/pca_plot_subgraph_{subgraph_id}_by_token.html")
    
    # Show if requested
    if show:
        fig.show()
    
    return fig


def plot_feature_activations(pca_df, results, feature_list, subgraph_id, show=True, save_path=None):
    """Create heatmap of feature activations across examples."""
    import plotly.express as px
    
    # Get feature activations
    feature_activations = results.all_graph_feature_acts.cpu().numpy()
    
    # Limit to first 100 examples for better visualization
    max_examples = min(100, feature_activations.shape[0])
    feature_activations = feature_activations[:max_examples]
    
    # Create a heatmap
    fig = px.imshow(
        feature_activations,
        labels=dict(x="Feature Index", y="Example", color="Activation"),
        x=[f"Feature {i}" for i in feature_list[:feature_activations.shape[1]]],
        title=f"Feature Activations for Subgraph {subgraph_id}",
        color_continuous_scale="Viridis",
    )
    
    # Save if requested
    if save_path:
        fig.write_image(f"{save_path}/feature_activations_subgraph_{subgraph_id}.png")
        fig.write_html(f"{save_path}/feature_activations_subgraph_{subgraph_id}.html")
    
    # Show if requested
    if show:
        fig.show()
    
    return fig


def plot_pca_by_feature_activation(pca_df, results, feature_list, subgraph_id, show=True, save_path=None):
    """Create PCA plots colored by top activated feature."""
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Function to get the active feature for each data point
    def get_active_feature(idx):
        activations = results.all_graph_feature_acts[idx].cpu().numpy()
        if np.all(activations == 0):
            return "None"
        active_idx = np.argmax(activations)
        return str(feature_list[active_idx])
    
    # Add column for the active feature
    pca_df["active_feature"] = [get_active_feature(i) for i in range(len(pca_df))]
    
    # Create color map
    unique_features = pca_df["active_feature"].unique()
    color_scale = px.colors.qualitative.Dark24
    color_map = {feature: color_scale[i % len(color_scale)] for i, feature in enumerate(unique_features)}
    
    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("PC1 vs PC2", "PC1 vs PC3", "PC2 vs PC3")
    )
    
    # Add traces for each projection
    for i, (x, y) in enumerate([("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")]):
        for feature in unique_features:
            mask = pca_df["active_feature"] == feature
            fig.add_trace(
                go.Scatter(
                    x=pca_df[mask][x],
                    y=pca_df[mask][y],
                    mode="markers",
                    marker=dict(color=color_map[feature]),
                    name=feature,
                    text=[
                        f"Token: {t}<br>Context: {c}<br>Active Feature: {f}"
                        for t, c, f in zip(
                            pca_df[mask]["tokens"],
                            pca_df[mask]["context"],
                            pca_df[mask]["active_feature"],
                        )
                    ],
                    hoverinfo="text",
                    showlegend=(i == 0),  # Only show legend for the first subplot
                ),
                row=1,
                col=i+1,
            )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1800,
        title_text=f"PCA Analysis by Active Feature - Subgraph {subgraph_id}",
        legend_title_text="Active Feature",
    )
    
    # Save if requested
    if save_path:
        fig.write_image(f"{save_path}/pca_plot_subgraph_{subgraph_id}_by_feature.png")
        fig.write_html(f"{save_path}/pca_plot_subgraph_{subgraph_id}_by_feature.html")
    
    # Show if requested
    if show:
        fig.show()
    
    return fig


def analyze_subgraph(
    sae_path: str,
    cooc_dir: str,
    output_dir: str,
    subgraph_id: int,
    activation_threshold: float = 1.5,
    n_batches: int = 10,
    show_plots: bool = True,
    save_data: bool = True,
    remove_special_tokens: bool = True,
    max_examples: int = 10000,
):
    """
    Analyze a specific subgraph with PCA
    
    Args:
        sae_path: Path to the SAE checkpoint
        cooc_dir: Directory containing co-occurrence data
        output_dir: Directory to save results
        subgraph_id: ID of the subgraph to analyze
        activation_threshold: Threshold for feature activation
        n_batches: Number of batches to process for reconstruction
        show_plots: Whether to display plots
        save_data: Whether to save data and plots
        remove_special_tokens: Whether to remove special tokens from processing
        max_examples: Maximum number of examples to process
    """
    # Set up directories and logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = pj(output_dir, f"subgraph_{subgraph_id}_{timestamp}")
    paths = create_directories(output_dir)
    setup_logging(pj(output_dir, "analysis.log"))
    
    device = set_device()
    logging.info(f"Using device: {device}")
    logging.info(f"Analyzing subgraph {subgraph_id} with activation threshold {activation_threshold}")
    
    # Load SAE model
    sae, cfg = load_matryoshka_sae(sae_path)
    
    # Load model
    model_name = cfg["model_name"]
    logging.info(f"Loading model {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    
    # Load node information
    sae_name = "matryoshka"  # Using the name from your cooccurrence generation
    threshold_str = str(activation_threshold).replace(".", "_")
    
    node_info_file = pj(cooc_dir, "graph_results", sae_name, "dataframes", f"{sae_name}_node_info_{threshold_str}.csv")
    if os.path.exists(node_info_file):
        node_df = pd.read_csv(node_info_file)
        logging.info(f"Loaded node info from {node_info_file}")
    else:
        # Try loading from co-occurrence directory directly
        node_df = load_node_info(cooc_dir, sae_name, activation_threshold)
    
    # Filter for the specific subgraph
    subgraph_nodes = node_df.query(f"subgraph_id == {subgraph_id}")["node_id"].tolist()
    
    if not subgraph_nodes:
        logging.error(f"No nodes found for subgraph {subgraph_id}")
        return None
    
    logging.info(f"Found {len(subgraph_nodes)} nodes in subgraph {subgraph_id}")
    
    # Set up activation store
    activation_store = ActivationsStore(model, cfg)
    
    # Generate and save pickle data
    pickle_file = pj(paths["data"], f"pca_data_subgraph_{subgraph_id}.pkl")
    
    if os.path.exists(pickle_file) and not save_data:
        logging.info(f"Loading existing data from {pickle_file}")
        data = load_data_from_pickle(pickle_file)
    else:
        logging.info("Generating new data")
        data = generate_data(
            model,
            sae,
            activation_store,
            subgraph_nodes,
            n_batches,
            decoder=True,
            remove_special_tokens=remove_special_tokens,
            device=device,
            max_examples=max_examples,
        )
        
        if save_data:
            save_data_to_pickle(data, pickle_file)
    
    results = data["results"]
    pca_df = data["pca_df"]
    pca = data["pca"]
    pca_decoder = data["pca_decoder"]
    pca_decoder_df = data["pca_decoder_df"]
    
    # Generate plots
    if pca_df is not None and not pca_df.empty:
        # Plot PCA by token
        plot_token_pca(
            pca_df, 
            subgraph_id, 
            title="PCA Analysis by Token",
            show=show_plots, 
            save_path=paths["plots"] if save_data else None
        )
        
        # Plot feature activations
        plot_feature_activations(
            pca_df,
            results,
            subgraph_nodes,
            subgraph_id,
            show=show_plots,
            save_path=paths["plots"] if save_data else None
        )
        
        # Plot PCA by feature activation
        plot_pca_by_feature_activation(
            pca_df,
            results,
            subgraph_nodes,
            subgraph_id,
            show=show_plots,
            save_path=paths["plots"] if save_data else None
        )
        
        # Plot PCA of decoder weights if available
        if pca_decoder is not None and pca_decoder_df is not None:
            plot_decoder_pca(
                pca_decoder_df,
                subgraph_id,
                show=show_plots,
                save_path=paths["plots"] if save_data else None
            )
        
        logging.info("All plots generated successfully")
    else:
        logging.warning("No data available for plotting. Check if enough examples were found.")
    
    return data


def plot_decoder_pca(pca_decoder_df, subgraph_id, show=True, save_path=None):
    """Create a scatter plot visualization of PCA results for decoder weights."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("PC1 vs PC2", "PC1 vs PC3", "PC2 vs PC3")
    )
    
    # Add traces for each projection
    fig.add_trace(
        go.Scatter(x=pca_decoder_df["PC1"], y=pca_decoder_df["PC2"], mode="markers"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=pca_decoder_df["PC1"], y=pca_decoder_df["PC3"], mode="markers"),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=pca_decoder_df["PC2"], y=pca_decoder_df["PC3"], mode="markers"),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1800,
        title_text=f"PCA of Decoder Weights - Subgraph {subgraph_id}",
        showlegend=False,
    )
    
    # Save if requested
    if save_path:
        fig.write_image(f"{save_path}/decoder_pca_subgraph_{subgraph_id}.png")
        fig.write_html(f"{save_path}/decoder_pca_subgraph_{subgraph_id}.html")
    
    # Show if requested
    if show:
        fig.show()
    
    return fig


def plot_3d_pca(pca_df, subgraph_id, color_by="token", show=True, save_path=None):
    """Create a 3D scatter plot visualization of PCA results."""
    import plotly.express as px
    
    if color_by == "token":
        fig = px.scatter_3d(
            pca_df, 
            x="PC1", 
            y="PC2", 
            z="PC3",
            color="tokens",
            hover_data=["context"],
            title=f"3D PCA Analysis by Token - Subgraph {subgraph_id}"
        )
    elif color_by == "active_feature":
        if "active_feature" not in pca_df.columns:
            # This would need to be calculated first
            return None
        fig = px.scatter_3d(
            pca_df, 
            x="PC1", 
            y="PC2", 
            z="PC3",
            color="active_feature",
            hover_data=["tokens", "context"],
            title=f"3D PCA Analysis by Active Feature - Subgraph {subgraph_id}"
        )
    
    # Update marker size
    fig.update_traces(marker=dict(size=5))
    
    # Save if requested
    if save_path:
        fig.write_html(f"{save_path}/pca_3d_subgraph_{subgraph_id}_{color_by}.html")
    
    # Show if requested
    if show:
        fig.show()
    
    return fig


def analyze_features_with_pca(results_dir, sae_path, output_dir):
    """Analyze all significant subgraphs with PCA."""
    # Set device
    device = set_device()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = pj(output_dir, f"matryoshka_pca_analysis_{timestamp}")
    paths = create_directories(output_dir)
    setup_logging(pj(output_dir, "full_analysis.log"))
    
    # Load SAE
    sae, cfg = load_matryoshka_sae(sae_path)
    
    # Load node information to find important subgraphs
    activation_threshold = 1.5
    threshold_str = str(activation_threshold).replace(".", "_")
    sae_name = "matryoshka"  # Using the name from your cooccurrence generation
    
    # Try to load node info from results dir
    node_info_file = pj(results_dir, "graph_results", sae_name, "dataframes", f"{sae_name}_node_info_{threshold_str}.csv")
    if os.path.exists(node_info_file):
        node_df = pd.read_csv(node_info_file)
        logging.info(f"Loaded node info from {node_info_file}")
    else:
        # Try loading from co-occurrence directory directly
        node_df = load_node_info(results_dir, sae_name, activation_threshold)
    
    # Find significant subgraphs (those with at least 10 nodes)
    subgraph_sizes = node_df.query("subgraph_id != 'Not in subgraph'").groupby("subgraph_id").size()
    significant_subgraphs = subgraph_sizes[subgraph_sizes >= 10].index.tolist()
    
    if not significant_subgraphs:
        logging.error("No significant subgraphs found")
        return
    
    logging.info(f"Found {len(significant_subgraphs)} significant subgraphs")
    
    # Analyze each significant subgraph
    for subgraph_id in significant_subgraphs[:5]:  # Limit to first 5 for testing
        try:
            logging.info(f"Analyzing subgraph {subgraph_id}")
            analyze_subgraph(
                sae_path=sae_path,
                cooc_dir=results_dir,
                output_dir=output_dir,
                subgraph_id=subgraph_id,
                activation_threshold=activation_threshold,
                n_batches=5,  # Reduced for full analysis
                show_plots=False,  # Don't show plots in batch mode
                save_data=True,
                remove_special_tokens=True,
                max_examples=5000,  # Reduced for full analysis
            )
        except Exception as e:
            logging.error(f"Error analyzing subgraph {subgraph_id}: {str(e)}")
    
    logging.info("Full analysis completed")


def main():
    """Main function to run the script."""
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description="Generate PCA visualizations for a Matryoshka SAE model")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to the SAE checkpoint")
    parser.add_argument("--cooc_dir", type=str, required=True, help="Directory containing co-occurrence data")
    parser.add_argument("--output_dir", type=str, default="./matryoshka_pca_results", help="Directory to save results")
    parser.add_argument("--subgraph_id", type=int, help="ID of the subgraph to analyze (if not specified, will analyze all significant subgraphs)")
    parser.add_argument("--threshold", type=float, default=1.5, help="Activation threshold")
    parser.add_argument("--n_batches", type=int, default=10, help="Number of batches to process")
    parser.add_argument("--no_show_plots", action="store_true", help="Don't display plots")
    parser.add_argument("--no_save_data", action="store_true", help="Don't save data and plots")
    parser.add_argument("--include_special_tokens", action="store_true", help="Include special tokens in processing")
    parser.add_argument("--max_examples", type=int, default=10000, help="Maximum number of examples to process")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.subgraph_id is not None:
        # Analyze a specific subgraph
        analyze_subgraph(
            sae_path=args.sae_path,
            cooc_dir=args.cooc_dir,
            output_dir=args.output_dir,
            subgraph_id=args.subgraph_id,
            activation_threshold=args.threshold,
            n_batches=args.n_batches,
            show_plots=not args.no_show_plots,
            save_data=not args.no_save_data,
            remove_special_tokens=not args.include_special_tokens,
            max_examples=args.max_examples,
        )
    else:
        # Analyze all significant subgraphs
        analyze_features_with_pca(
            results_dir=args.cooc_dir,
            sae_path=args.sae_path,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()