import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from matryoshka_cooc.config import get_default_cfg, post_init_cfg
from matryoshka_cooc.sae import GlobalBatchTopKMatryoshkaSAE


def load_matryoshka_sae(
    checkpoint_path: str, cfg_dict: dict | None = None
) -> tuple[GlobalBatchTopKMatryoshkaSAE, dict]:
    """Load a trained matryoshka SAE model"""
    if cfg_dict is None:
        cfg = get_default_cfg()
        # Override default config with visualization-specific settings
    else:
        cfg = cfg_dict
        
    cfg.update(
            {
                "model_name": "gpt2-small",
                "layer": 0,
                "site": "resid_pre",
                "act_size": 768,
                "dict_size": 768 * 8,
                "sae_type": "global-matryoshka-topk",
                "group_sizes": [768, 768, 768 * 2, 768 * 4],
                "device": "cuda"
                if torch.cuda.is_available()
                else "cpu",  # Default to CPU if CUDA not available
                "top_k": 32,
                "lr": 3e-4,
                "seed": 42,
                "dtype": torch.float32,
                "dataset_path": "Skylion007/openwebtext",
                "model_batch_size": 32,
                "batch_size": 512,
                "seq_len": 1024,
                "num_batches_in_buffer": 10,
            }
        )
    cfg = post_init_cfg(cfg)

    # Load trained weights and config
    state_dict = torch.load(checkpoint_path, map_location=cfg["device"])
    
    # If this is a checkpoint file, it might contain the config in a specific format
    if isinstance(checkpoint_path, str) and checkpoint_path.endswith('.pt'):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                checkpoint_cfg = json.load(f)
                # Convert group_sizes from string to list if needed
                if isinstance(checkpoint_cfg.get('group_sizes'), str):
                    checkpoint_cfg['group_sizes'] = json.loads(checkpoint_cfg['group_sizes'])
                cfg.update(checkpoint_cfg)

    # Create model with the same architecture
    sae = GlobalBatchTopKMatryoshkaSAE(cfg)

    # Load trained weights
    sae.load_state_dict(state_dict)
    sae = sae.to(cfg["device"])

    return sae, cfg


def analyze_feature_usage(
    sae: GlobalBatchTopKMatryoshkaSAE, dataset_loader, n_batches: int = 10
) -> pd.DataFrame:
    """Analyze how features are used in the matryoshka SAE"""
    # Get device from SAE
    device = next(sae.parameters()).device

    # Initialize statistics containers
    feature_activations = torch.zeros(sae.config["dict_size"], device=device)
    feature_magnitudes = torch.zeros(sae.config["dict_size"], device=device)

    group_sizes = sae.group_sizes
    group_indices = sae.group_indices

    # Process batches
    sae.eval()  # Set to evaluation mode
    with torch.no_grad():
        for _ in tqdm(range(n_batches), desc="Analyzing features"):
            batch = dataset_loader.next_batch()
            # Move batch to the same device as SAE
            batch = batch.to(device)

            # Forward pass through the SAE
            outputs = sae(batch)
            # Get feature activations
            acts = outputs["feature_acts"]

            # Update statistics
            feature_activations += (acts > 0).float().sum(dim=0)
            feature_magnitudes += acts.abs().sum(dim=0)

    # Convert to average per batch
    feature_activations = feature_activations.cpu().numpy() / n_batches
    feature_magnitudes = feature_magnitudes.cpu().numpy() / n_batches

    # Create group labels
    group_labels = []
    for i in range(len(group_sizes)):
        start = group_indices[i]
        end = group_indices[i + 1]
        size = group_sizes[i]
        label = f"Group {i} (size: {size})"
        group_labels.extend([label] * (end - start))

    # Create a DataFrame for easier analysis
    df = pd.DataFrame(
        {
            "Feature": range(sae.config["dict_size"]),
            "Activation_Frequency": feature_activations,
            "Average_Magnitude": feature_magnitudes,
            "Group": group_labels,
        }
    )

    return df


def visualize_feature_stats(stats_df: pd.DataFrame):
    """Create visualizations for feature statistics"""
    # Set style
    # # plt.style.use("seaborn")
    # sns.set_palette("husl")

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Activation frequency by group
    sns.boxplot(x="Group", y="Activation_Frequency", data=stats_df, ax=axes[0, 0])
    axes[0, 0].set_title("Feature Activation Frequency by Group")
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)

    # 2. Average magnitude by group
    sns.boxplot(x="Group", y="Average_Magnitude", data=stats_df, ax=axes[0, 1])
    axes[0, 1].set_title("Feature Activation Magnitude by Group")
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

    # 3. Histogram of activation frequencies
    sns.histplot(
        data=stats_df,
        x="Activation_Frequency",
        hue="Group",
        element="step",
        ax=axes[1, 0],
        common_norm=False,
        stat="density",
    )
    axes[1, 0].set_title("Distribution of Activation Frequencies")

    # 4. Scatter plot of activation frequency vs magnitude
    sns.scatterplot(
        data=stats_df,
        x="Activation_Frequency",
        y="Average_Magnitude",
        hue="Group",
        alpha=0.6,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Activation Frequency vs Magnitude")

    plt.tight_layout()
    plt.savefig("matryoshka_feature_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    return fig


def analyze_decoder_weights(sae: GlobalBatchTopKMatryoshkaSAE):
    """Analyze the decoder weights of the matryoshka SAE"""
    # Get group information
    group_sizes = sae.group_sizes
    group_indices = sae.group_indices

    # Analysis containers
    group_norms = []

    # Compute weight norms for each group
    with torch.no_grad():
        for i in range(len(group_sizes)):
            start = group_indices[i]
            end = group_indices[i + 1]

            # Extract group weights
            group_weights = sae.W_dec[start:end]

            # Compute norms
            weight_norms = torch.norm(group_weights, dim=1)
            group_norms.append(weight_norms.cpu().numpy())

    # Create visualization
    # plt.style.use("seaborn")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot distribution of weight norms for each group
    positions = np.arange(len(group_sizes))
    bp = ax.boxplot(group_norms, positions=positions, patch_artist=True)

    # Customize appearance
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(group_sizes)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    # Add labels
    ax.set_title("Distribution of Decoder Weight Norms by Group")
    ax.set_xlabel("Group")
    ax.set_ylabel("Weight Norm")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"Group {i}\n(size: {size})" for i, size in enumerate(group_sizes)]
    )

    plt.tight_layout()
    plt.savefig("matryoshka_weight_norms.png", dpi=300, bbox_inches="tight")
    plt.close()

    return fig, group_norms


def main():
    # Determine device - prefer CUDA, fallback to CPU (skip MPS for now)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and SAE
    model = HookedTransformer.from_pretrained("gpt2-small")
    model = model.to(device)

    # Path to your trained SAE checkpoint
    checkpoint_path = "checkpoints/gpt2-small_blocks.0.hook_resid_pre_6144_global-matryoshka-topk_32_0.0003_final.pt"

    # Load SAE with device specified in config
    cfg_dict = get_default_cfg()
    cfg_dict["device"] = device
    sae, cfg = load_matryoshka_sae(checkpoint_path, cfg_dict)

    # Set up activation store for analysis
    from matryoshka_cooc.activation_store import ActivationsStore

    activations_store = ActivationsStore(model, cfg)

    # Analyze feature usage
    stats_df = analyze_feature_usage(sae, activations_store)

    # Visualize statistics
    visualize_feature_stats(stats_df)

    # Analyze decoder weights
    analyze_decoder_weights(sae)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
