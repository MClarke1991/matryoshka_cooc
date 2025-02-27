import os
import pickle
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
from sae_lens import ActivationsStore as SaeLensStore
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from matryoshka_cooc.activation_store import ActivationsStore
from matryoshka_cooc.config import get_default_cfg, post_init_cfg
from matryoshka_cooc.sae import GlobalBatchTopKMatryoshkaSAE


def set_device() -> str:
    """Set device based on availability"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_matryoshka_sae(checkpoint_path: str | None = None) -> tuple[GlobalBatchTopKMatryoshkaSAE, dict[str, Any]]:
    """
    Load a trained matryoshka SAE model or initialize a new one with the specified configuration
    
    Args:
        checkpoint_path: Path to the checkpoint file, or None to use random initialization
        
    Returns:
        Tuple of SAE model and configuration
    """
    device = set_device()

    # Configuration for GPT2-small
    cfg = get_default_cfg()
    cfg["model_name"] = "gpt2-small"
    cfg["layer"] = 0
    cfg["site"] = "resid_pre"
    cfg["act_size"] = 768
    cfg["device"] = device

    # Matryoshka-specific configurations
    cfg["sae_type"] = "global-matryoshka-topk"
    cfg["dict_size"] = 768 * 64  # Total dictionary size
    cfg["top_k"] = 32
    cfg["group_sizes"] = [768, 768, 768 * 2, 768 * 4, 768 * 8, 768 * 16, 768 * 32]

    # Update config with derived values
    cfg = post_init_cfg(cfg)

    # Create model with the same architecture
    sae = GlobalBatchTopKMatryoshkaSAE(cfg)

    # Load trained weights if provided
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        sae.load_state_dict(state_dict)
        print(f"Loaded Matryoshka SAE from {checkpoint_path}")
    else:
        print("Using randomly initialized Matryoshka SAE")

    return sae, cfg


def load_resjb_sae() -> SAE:
    """
    Load res-jb SAE for block 0 of GPT2-small
    
    Returns:
        Loaded SAE model
    """
    device = set_device()

    # Load res-jb SAE using sae_lens
    print("Loading res-jb SAE for block 0...")
    sae, _, _ = SAE.from_pretrained(
        release="gpt2-small-res-jb", 
        sae_id="blocks.0.hook_resid_pre", 
        device=device
    )

    return sae


def create_activation_store(model: HookedTransformer, cfg: dict, sae_type: str = "matryoshka", sae: SAE | None = None) -> ActivationsStore | SaeLensStore:
    """
    Create appropriate activation store based on SAE type

    Args:
        model: The language model
        cfg: Configuration dictionary
        sae_type: Type of SAE ("matryoshka" or "resjb")
        sae: The SAE model (required for res-jb)

    Returns:
        Activation store instance
    """
    if sae_type == "matryoshka":
        return ActivationsStore(model, cfg)
    else:
        # For res-jb SAE, use the sae_lens activation store
        if sae is None:
            raise ValueError("SAE must be provided for res-jb SAE")
        return SaeLensStore.from_sae(
            model=model,
            sae=sae,
            streaming=True,
            store_batch_size_prompts=8,
            train_batch_size_tokens=4096,
            n_batches_in_buffer=8,
        )


def compute_normalized_cooccurrence_matrix(
    sae: GlobalBatchTopKMatryoshkaSAE | SAE,
    activation_store: ActivationsStore | SaeLensStore,
    dict_size: int,
    activation_threshold: float,
    n_batches: int = 100,
    batch_size: int = 1000,
    is_matryoshka: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute normalized co-occurrence matrix for all features
    
    Args:
        sae: The sparse autoencoder
        activation_store: Store to get batches of activations
        dict_size: Size of the SAE dictionary
        activation_threshold: Threshold for considering a feature active
        n_batches: Number of batches to process
        batch_size: Size of mini-batches for processing large matrices
        is_matryoshka: Whether the SAE is a matryoshka model
        
    Returns:
        Tuple of (co-occurrence matrix, feature activations)
    """
    device = sae.W_dec.device
    
    # Initialize co-occurrence matrix and feature activations
    cooccurrence = torch.zeros((dict_size, dict_size), device=device)
    feature_activations = torch.zeros(dict_size, device=device)
    
    print(f"Computing co-occurrence matrix for SAE with {dict_size} features")
    print(f"Using device: {device}, threshold: {activation_threshold}")
    
    # Process batches
    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        # Get batch of activations
        batch = activation_store.next_batch().to(device)
        
        # Encode with SAE
        with torch.no_grad():
            feature_acts = sae.encode(batch)
            
            # Flatten batch dimension if needed
            if len(feature_acts.shape) > 2:
                feature_acts = feature_acts.reshape(-1, feature_acts.shape[-1])
                
            # Apply threshold
            binary_acts = (feature_acts > activation_threshold).float()
            
            # Update feature activations count
            feature_activations += binary_acts.sum(dim=0)
            
            # Process in mini-batches to avoid OOM issues
            for i in range(0, binary_acts.shape[0], batch_size):
                mini_batch = binary_acts[i:i+batch_size]
                # Update co-occurrence matrix
                mini_batch_cooc = torch.matmul(mini_batch.t(), mini_batch)
                cooccurrence += mini_batch_cooc
                
    # Normalize co-occurrence matrix (using outer product method)
    # This computes P(j|i) = P(i,j) / P(i)
    feature_activations_expanded = feature_activations.unsqueeze(1)
    norm_denom = torch.max(feature_activations_expanded, feature_activations_expanded.t())
    norm_cooccurrence = cooccurrence / (norm_denom + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Set diagonal to 1
    norm_cooccurrence.fill_diagonal_(1.0)
    
    return norm_cooccurrence, feature_activations


def save_cooccurrence_data(
    norm_cooccurrence: torch.Tensor,
    feature_activations: torch.Tensor,
    output_dir: str,
    sae_name: str,
    activation_threshold: float
) -> None:
    """
    Save co-occurrence matrix and feature activations to disk
    
    Args:
        norm_cooccurrence: Normalized co-occurrence matrix
        feature_activations: Feature activation counts
        output_dir: Directory to save output files
        sae_name: Name of the SAE (for filename)
        activation_threshold: Activation threshold used (for filename)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy for saving
    norm_cooccurrence_np = norm_cooccurrence.cpu().numpy()
    feature_activations_np = feature_activations.cpu().numpy()
    
    # Create filenames
    threshold_str = str(activation_threshold).replace(".", "_")
    
    cooccurrence_file = os.path.join(output_dir, f"{sae_name}_cooccurrence_{threshold_str}.npz")
    activations_file = os.path.join(output_dir, f"{sae_name}_activations_{threshold_str}.npz")
    
    # Save files
    np.savez_compressed(cooccurrence_file, norm_cooccurrence_np)
    np.savez_compressed(activations_file, feature_activations_np)
    
    print(f"Saved co-occurrence matrix to {cooccurrence_file}")
    print(f"Saved feature activations to {activations_file}")


def remove_self_loops(matrix: torch.Tensor) -> torch.Tensor:
    """
    Remove self-loops from co-occurrence matrix by setting diagonal to 0
    
    Args:
        matrix: Co-occurrence matrix
        
    Returns:
        Matrix with diagonal elements set to zero
    """
    result = matrix.clone()
    result.fill_diagonal_(0.0)
    return result


def calculate_jaccard_matrix(
    norm_cooccurrence: torch.Tensor,
    feature_activations: torch.Tensor,
    batch_size: int = 1000
) -> torch.Tensor:
    """
    Calculate Jaccard similarity matrix from co-occurrence matrix
    
    Args:
        norm_cooccurrence: Normalized co-occurrence matrix
        feature_activations: Feature activation counts
        batch_size: Size of mini-batches for processing large matrices
        
    Returns:
        Jaccard similarity matrix
    """
    device = norm_cooccurrence.device
    n_features = norm_cooccurrence.shape[0]
    jaccard_matrix = torch.zeros((n_features, n_features), device=device)
    
    # Remove self-loops for calculations
    cooccurrence_no_self = remove_self_loops(norm_cooccurrence)
    
    # Process in batches to avoid OOM
    for i in tqdm(range(0, n_features, batch_size), desc="Calculating Jaccard matrix"):
        end = min(i + batch_size, n_features)
        batch = cooccurrence_no_self[i:end, :]
        
        # Calculate intersection (co-occurrence)
        intersection = batch
        
        # Calculate union
        # Union = A + B - intersection
        # In our case, we're directly calculating with the normalized co-occurrence values
        union = feature_activations[i:end].unsqueeze(1) + feature_activations.unsqueeze(0) - intersection
        
        # Calculate Jaccard similarity: intersection / union
        batch_jaccard = intersection / (union + 1e-10)  # Add small epsilon to avoid division by zero
        
        # Handle infinities and NaNs
        batch_jaccard = torch.nan_to_num(batch_jaccard, nan=0.0, posinf=1.0, neginf=0.0)
        
        jaccard_matrix[i:end, :] = batch_jaccard
    
    # Set diagonal to 1
    jaccard_matrix.fill_diagonal_(1.0)
    
    return jaccard_matrix


def generate_cooccurrence(
    sae: GlobalBatchTopKMatryoshkaSAE | SAE,
    activation_store: ActivationsStore | SaeLensStore,
    dict_size: int,
    output_dir: str,
    sae_name: str,
    activation_thresholds: list[float],
    n_batches: int = 100,
    compute_jaccard: bool = True,
    is_matryoshka: bool = False,
) -> None:
    """
    Generate and save co-occurrence and jaccard matrices for all specified thresholds
    
    Args:
        sae: The sparse autoencoder
        activation_store: Store to get batches of activations
        dict_size: Size of the SAE dictionary
        output_dir: Directory to save output files
        sae_name: Name of the SAE (for filename)
        activation_thresholds: List of activation thresholds to use
        n_batches: Number of batches to process
        compute_jaccard: Whether to compute Jaccard similarity matrices
        is_matryoshka: Whether the SAE is a matryoshka model
    """
    for threshold in activation_thresholds:
        print(f"\nComputing co-occurrence for {sae_name} with threshold {threshold}")
        
        # Compute normalized co-occurrence matrix
        start_time = time.time()
        norm_cooccurrence, feature_activations = compute_normalized_cooccurrence_matrix(
            sae=sae,
            activation_store=activation_store,
            dict_size=dict_size,
            activation_threshold=threshold,
            n_batches=n_batches,
            is_matryoshka=is_matryoshka
        )
        print(f"Co-occurrence computation time: {time.time() - start_time:.2f} seconds")
        
        # Save co-occurrence data
        save_cooccurrence_data(
            norm_cooccurrence=norm_cooccurrence,
            feature_activations=feature_activations,
            output_dir=output_dir,
            sae_name=sae_name,
            activation_threshold=threshold
        )
        
        # Compute and save Jaccard matrix if requested
        if compute_jaccard:
            print(f"Computing Jaccard similarity matrix for {sae_name} with threshold {threshold}")
            start_time = time.time()
            jaccard_matrix = calculate_jaccard_matrix(norm_cooccurrence, feature_activations)
            print(f"Jaccard computation time: {time.time() - start_time:.2f} seconds")
            
            # Save Jaccard matrix
            threshold_str = str(threshold).replace(".", "_")
            jaccard_file = os.path.join(output_dir, f"{sae_name}_jaccard_{threshold_str}.npz")
            np.savez_compressed(jaccard_file, jaccard_matrix.cpu().numpy())
            print(f"Saved Jaccard matrix to {jaccard_file}")
            
        # Clear GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def save_config_info(output_dir: str, matryoshka_cfg: dict | None = None, resjb_info: dict | None = None) -> None:
    """
    Save configuration information about the SAEs
    
    Args:
        output_dir: Directory to save configuration file
        matryoshka_cfg: Configuration dictionary for matryoshka SAE
        resjb_info: Information dictionary for res-jb SAE
    """
    os.makedirs(output_dir, exist_ok=True)
    
    config_info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": set_device(),
    }
    
    if matryoshka_cfg:
        config_info["matryoshka"] = {
            "model_name": matryoshka_cfg.get("model_name"),
            "layer": matryoshka_cfg.get("layer"),
            "site": matryoshka_cfg.get("site"),
            "dict_size": matryoshka_cfg.get("dict_size"),
            "top_k": matryoshka_cfg.get("top_k"),
            "group_sizes": matryoshka_cfg.get("group_sizes"),
        }
    
    if resjb_info:
        config_info["resjb"] = resjb_info
    
    # Save as pickle
    with open(os.path.join(output_dir, "config_info.pkl"), "wb") as f:
        pickle.dump(config_info, f)
        
    # Also save as text for easy reading
    with open(os.path.join(output_dir, "config_info.txt"), "w") as f:
        f.write("=== Co-occurrence Generation Configuration ===\n\n")
        f.write(f"Timestamp: {config_info['timestamp']}\n")
        f.write(f"Device: {config_info['device']}\n\n")
        
        if matryoshka_cfg:
            f.write("=== Matryoshka SAE ===\n")
            for key, value in config_info["matryoshka"].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        if resjb_info:
            f.write("=== Res-JB SAE ===\n")
            for key, value in config_info["resjb"].items():
                f.write(f"{key}: {value}\n")


def main() -> None:
    # Set parameters
    output_dir = "cooccurrence_results"
    n_batches = 50  # Number of batches to process
    activation_thresholds = [0.0, 1.5]  # Thresholds to try
    
    device = set_device()
    print(f"Using device: {device}")
    
    # Load model
    model = HookedTransformer.from_pretrained("gpt2-small").to(device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize SAE tracking
    matryoshka_loaded = False
    resjb_loaded = False
    
    # Try to load Matryoshka SAE
    matryoshka_path = "checkpoints/gpt2-small_blocks.0.hook_resid_pre_24576_global-matryoshka-topk_32_0.0003_final.pt"
    try:
        matryoshka_sae, matryoshka_cfg = load_matryoshka_sae(matryoshka_path)
        matryoshka_loaded = True
        print("Successfully loaded Matryoshka SAE")
    except FileNotFoundError:
        print(f"Could not find matryoshka checkpoint at {matryoshka_path}")
        matryoshka_loaded = False
    
    # Try to load res-jb SAE
    try:
        resjb_sae = load_resjb_sae()
        resjb_loaded = True
        print("Successfully loaded res-jb SAE")
        
        # Extract necessary information
        resjb_info = {
            "model_name": "gpt2-small",
            "layer": 0,
            "site": "blocks.0.hook_resid_pre", 
            "dict_size": resjb_sae.cfg.d_sae,
            "release": "gpt2-small-res-jb",
        }
    except Exception as e:
        print(f"Could not load res-jb SAE: {e}")
        resjb_loaded = False
    
    # Save configuration information
    save_config_info(
        output_dir=output_dir,
        matryoshka_cfg=matryoshka_cfg if matryoshka_loaded else None,
        resjb_info=resjb_info if resjb_loaded else None
    )
    
    # Process Matryoshka SAE if loaded
    if matryoshka_loaded:
        # Set up activation store
        matryoshka_activation_store = create_activation_store(model, matryoshka_cfg, "matryoshka")
        
        # Generate co-occurrence
        generate_cooccurrence(
            sae=matryoshka_sae,
            activation_store=matryoshka_activation_store,
            dict_size=matryoshka_cfg["dict_size"],
            output_dir=output_dir,
            sae_name="matryoshka",
            activation_thresholds=activation_thresholds,
            n_batches=n_batches,
            is_matryoshka=True
        )
    
    # Process res-jb SAE if loaded
    if resjb_loaded:
        # Set up configuration for activation store
        resjb_cfg = get_default_cfg()
        resjb_cfg["model_name"] = "gpt2-small"
        resjb_cfg["layer"] = 0
        resjb_cfg["site"] = "blocks.0.hook_resid_pre"
        resjb_cfg["act_size"] = 768
        resjb_cfg["device"] = device
        resjb_cfg = post_init_cfg(resjb_cfg)
        
        # Set up activation store
        resjb_activation_store = create_activation_store(model, resjb_cfg, "resjb", resjb_sae)
        
        # Generate co-occurrence
        generate_cooccurrence(
            sae=resjb_sae,
            activation_store=resjb_activation_store,
            dict_size=resjb_sae.cfg.d_sae,
            output_dir=output_dir,
            sae_name="resjb",
            activation_thresholds=activation_thresholds,
            n_batches=n_batches,
            is_matryoshka=False
        )
    
    print("Co-occurrence generation complete!")


if __name__ == "__main__":
    main()