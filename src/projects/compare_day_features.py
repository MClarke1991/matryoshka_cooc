import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sae_lens import SAE
from scipy.spatial.distance import squareform
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from matryoshka_cooc.activation_store import ActivationsStore
from matryoshka_cooc.config import get_default_cfg, post_init_cfg
from matryoshka_cooc.sae import GlobalBatchTopKMatryoshkaSAE, TopKSAE

# Days of the week for search
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DAYS_CAPS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
ALL_DAYS = DAYS + DAYS_CAPS


def set_device():
    """Set device based on availability"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def identify_day_features(sae, model, is_matryoshka=False):
    """
    Identify features that correspond to days of the week

    Args:
        sae: The sparse autoencoder model
        model: The language model
        is_matryoshka: Whether the SAE is a matryoshka model

    Returns:
        dict: Mapping from day name to list of feature indices
    """
    device = sae.W_dec.device

    # Get token indices for days of the week
    day_tokens = {}
    day_indices = {}

    for day in ALL_DAYS:
        # Get token ID
        token_id = model.tokenizer.encode(" " + day)[
            0
        ]  # Add space for better tokenization
        day_tokens[day] = token_id

    # Get embedding matrix
    W_E = model.W_E.to(device)

    # Calculate correlations between embeddings and decoder weights
    corrs = {}

    if is_matryoshka:
        decoder_weights = sae.W_dec
    else:
        decoder_weights = sae.W_dec

    # For each day, find features that have high correlation with the day token
    for day, token_id in day_tokens.items():
        # Get token embedding
        token_emb = W_E[token_id]

        # Calculate correlation with all feature decoders
        feature_corrs = torch.mv(decoder_weights, token_emb)

        # Normalize to get correlation coefficient
        norms = torch.norm(decoder_weights, dim=1) * torch.norm(token_emb)
        feature_corrs = feature_corrs / norms

        corrs[day] = feature_corrs

    # Find the top features for each day
    day_features = {}

    for day in ALL_DAYS:
        # Get top 5 features by correlation
        values, indices = torch.topk(corrs[day], 5)
        day_features[day] = {
            "indices": indices.cpu().numpy(),
            "corrs": values.detach().cpu().numpy(),
        }

    return day_features, corrs


def compute_feature_cooccurrence(
    sae, activations_store, feature_indices, n_batches=50, is_matryoshka=False
):
    """
    Compute co-occurrence matrix for specific features

    Args:
        sae: The sparse autoencoder
        activations_store: Store to get batches of activations
        feature_indices: List of feature indices to compute co-occurrence for
        n_batches: Number of batches to process
        is_matryoshka: Whether the SAE is a matryoshka model

    Returns:
        np.ndarray: Co-occurrence matrix
    """
    n_features = len(feature_indices)
    device = sae.W_dec.device

    # Convert to tensor on device
    feature_indices_tensor = torch.tensor(feature_indices, device=device)

    # Initialize co-occurrence matrix
    cooccurrence = torch.zeros((n_features, n_features), device=device)

    # Process batches
    for _ in tqdm(range(n_batches), desc="Computing co-occurrence"):
        # Get batch of activations
        batch = activations_store.next_batch()

        # Encode with SAE
        with torch.no_grad():
            feature_acts = sae.encode(batch)

            # Flatten the batch dimension
            if len(feature_acts.shape) > 2:
                feature_acts = feature_acts.reshape(-1, feature_acts.shape[-1])

            # Extract just the features we care about
            selected_acts = feature_acts[:, feature_indices_tensor]

            # Binarize
            binary_acts = (selected_acts > 0).float()

            # Update cooccurrence matrix
            batch_cooccurrence = torch.matmul(binary_acts.t(), binary_acts)
            cooccurrence += batch_cooccurrence

    # Normalize by diagonal (frequency of each feature)
    diag = torch.diag(cooccurrence)
    norm_cooccurrence = cooccurrence / torch.max(diag.unsqueeze(0), diag.unsqueeze(1))

    # Set diagonal to 1
    norm_cooccurrence.fill_diagonal_(1.0)

    return norm_cooccurrence.cpu().numpy()


def visualize_cooccurrence(cooccurrence, feature_labels, title):
    """
    Visualize co-occurrence matrix

    Args:
        cooccurrence: Co-occurrence matrix
        feature_labels: Labels for features
        title: Plot title
    """
    plt.figure(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        cooccurrence,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=feature_labels,
        yticklabels=feature_labels,
    )

    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    plt.show()

    # Create a graph visualization
    G = nx.Graph()

    # Add nodes
    for i, label in enumerate(feature_labels):
        G.add_node(i, label=label)

    # Add edges based on co-occurrence
    for i in range(len(feature_labels)):
        for j in range(i + 1, len(feature_labels)):
            if (
                cooccurrence[i, j] > 0.1
            ):  # Only add edges with significant co-occurrence
                G.add_edge(i, j, weight=cooccurrence[i, j])

    # Draw the graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)  # For reproducible layout

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue")

    # Draw edges with width proportional to weight
    weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7)

    # Draw labels
    nx.draw_networkx_labels(
        G, pos, {i: label for i, label in enumerate(feature_labels)}
    )

    plt.title(f"Co-occurrence Graph - {title}", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}_graph.png", dpi=300)
    plt.show()


def load_matryoshka_sae(checkpoint_path=None):
    """Load a trained matryoshka SAE model"""
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
    cfg["dict_size"] = 768 * 8
    cfg["top_k"] = 32
    cfg["group_sizes"] = [768, 768, 768 * 2, 768 * 4]

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


def load_resjb_sae():
    """Load res-jb SAE for block 0 of GPT2-small"""
    device = set_device()

    # Load res-jb SAE using sae_lens
    print("Loading res-jb SAE for block 0...")
    sae, _, _ = SAE.from_pretrained(
        release="gpt2-small-res-jb", sae_id="blocks.0.hook_resid_pre", device=device
    )

    return sae


def main():
    device = set_device()
    print(f"Using device: {device}")

    # Load GPT2-small
    model = HookedTransformer.from_pretrained("gpt2-small").to(device)

    # Load SAEs
    matryoshka_path = "checkpoints/gpt2-small_blocks.0.hook_resid_pre_6144_global-matryoshka-topk_32_0.0003_final.pt"

    # Try to load trained matryoshka SAE, if not available, use randomly initialized
    try:
        matryoshka_sae, matryoshka_cfg = load_matryoshka_sae(matryoshka_path)
        matryoshka_trained = True
    except FileNotFoundError:
        print(
            f"Could not find matryoshka checkpoint at {matryoshka_path}, using randomly initialized"
        )
        matryoshka_sae, matryoshka_cfg = load_matryoshka_sae()
        matryoshka_trained = False

    # Try to load res-jb SAE
    try:
        resjb_sae = load_resjb_sae()
        resjb_loaded = True
    except Exception as e:
        print(f"Could not load res-jb SAE: {e}")
        resjb_loaded = False

    # Set up activation stores
    matryoshka_activation_store = ActivationsStore(model, matryoshka_cfg)

    if resjb_loaded:
        # Create compatible config for resjb
        resjb_cfg = get_default_cfg()
        resjb_cfg["model_name"] = "gpt2-small"
        resjb_cfg["layer"] = 0
        resjb_cfg["site"] = "blocks.0"
        resjb_cfg["act_size"] = 768
        resjb_cfg["device"] = device
        resjb_cfg = post_init_cfg(resjb_cfg)

        resjb_activation_store = ActivationsStore(model, resjb_cfg)

    # Identify day features in both SAEs
    print("Identifying day-of-week features in Matryoshka SAE...")
    matryoshka_day_features, matryoshka_corrs = identify_day_features(
        matryoshka_sae, model, is_matryoshka=True
    )

    if resjb_loaded:
        print("Identifying day-of-week features in res-jb SAE...")
        resjb_day_features, resjb_corrs = identify_day_features(
            resjb_sae, model, is_matryoshka=False
        )

    # Print identified features
    print("\nIdentified day features in Matryoshka SAE:")
    for day in DAYS:
        indices = matryoshka_day_features[day]["indices"]
        corrs = matryoshka_day_features[day]["corrs"]
        print(f"{day.capitalize()}: Features {indices} (correlations: {corrs})")

    if resjb_loaded:
        print("\nIdentified day features in res-jb SAE:")
        for day in DAYS:
            indices = resjb_day_features[day]["indices"]
            corrs = resjb_day_features[day]["corrs"]
            print(f"{day.capitalize()}: Features {indices} (correlations: {corrs})")

    # Extract top feature for each day
    matryoshka_day_top_features = {
        day: int(matryoshka_day_features[day]["indices"][0]) for day in DAYS
    }

    if resjb_loaded:
        resjb_day_top_features = {
            day: int(resjb_day_features[day]["indices"][0]) for day in DAYS
        }

    # Compute co-occurrence
    print("\nComputing co-occurrence for Matryoshka SAE...")
    matryoshka_feature_indices = list(matryoshka_day_top_features.values())
    matryoshka_feature_labels = list(matryoshka_day_top_features.keys())

    matryoshka_cooccurrence = compute_feature_cooccurrence(
        matryoshka_sae,
        matryoshka_activation_store,
        matryoshka_feature_indices,
        n_batches=50,
        is_matryoshka=True,
    )

    if resjb_loaded:
        print("Computing co-occurrence for res-jb SAE...")
        resjb_feature_indices = list(resjb_day_top_features.values())
        resjb_feature_labels = list(resjb_day_top_features.keys())

        resjb_cooccurrence = compute_feature_cooccurrence(
            resjb_sae,
            resjb_activation_store,
            resjb_feature_indices,
            n_batches=50,
            is_matryoshka=False,
        )

    # Visualize co-occurrence
    print("Visualizing co-occurrence...")
    matryoshka_title = "Matryoshka SAE - Day Features Co-occurrence"
    matryoshka_title += " (Randomly Initialized)" if not matryoshka_trained else ""

    visualize_cooccurrence(
        matryoshka_cooccurrence,
        [day.capitalize() for day in matryoshka_feature_labels],
        matryoshka_title,
    )

    if resjb_loaded:
        visualize_cooccurrence(
            resjb_cooccurrence,
            [day.capitalize() for day in resjb_feature_labels],
            "Res-JB SAE - Day Features Co-occurrence",
        )

    # Compare co-occurrence patterns
    if resjb_loaded:
        print("\nComparing co-occurrence patterns:")

        # Calculate average co-occurrence excluding diagonal
        matryoshka_avg = (
            matryoshka_cooccurrence.sum() - matryoshka_cooccurrence.trace()
        ) / (matryoshka_cooccurrence.size - matryoshka_cooccurrence.shape[0])
        resjb_avg = (resjb_cooccurrence.sum() - resjb_cooccurrence.trace()) / (
            resjb_cooccurrence.size - resjb_cooccurrence.shape[0]
        )

        print(f"Matryoshka average co-occurrence: {matryoshka_avg:.4f}")
        print(f"Res-JB average co-occurrence: {resjb_avg:.4f}")

        # Calculate variance in co-occurrence
        matryoshka_var = np.var(squareform(matryoshka_cooccurrence))
        resjb_var = np.var(squareform(resjb_cooccurrence))

        print(f"Matryoshka co-occurrence variance: {matryoshka_var:.4f}")
        print(f"Res-JB co-occurrence variance: {resjb_var:.4f}")

        # Compare specific day pairs
        print("\nDay pair comparison:")
        day_pairs = [
            ("monday", "tuesday"),
            ("monday", "wednesday"),
            ("saturday", "sunday"),
            ("wednesday", "friday"),
        ]

        comparison_df = []

        for day1, day2 in day_pairs:
            idx1_m = matryoshka_feature_labels.index(day1)
            idx2_m = matryoshka_feature_labels.index(day2)
            m_cooc = matryoshka_cooccurrence[idx1_m, idx2_m]

            idx1_r = resjb_feature_labels.index(day1)
            idx2_r = resjb_feature_labels.index(day2)
            r_cooc = resjb_cooccurrence[idx1_r, idx2_r]

            comparison_df.append(
                {
                    "Day Pair": f"{day1.capitalize()}-{day2.capitalize()}",
                    "Matryoshka": m_cooc,
                    "Res-JB": r_cooc,
                    "Difference": m_cooc - r_cooc,
                }
            )

            print(
                f"{day1.capitalize()}-{day2.capitalize()}: Matryoshka = {m_cooc:.4f}, Res-JB = {r_cooc:.4f}, Diff = {m_cooc - r_cooc:.4f}"
            )

        # Create comparison plot
        comparison_df = pd.DataFrame(comparison_df)

        plt.figure(figsize=(10, 6))
        x = np.arange(len(comparison_df))
        width = 0.35

        plt.bar(
            x - width / 2, comparison_df["Matryoshka"], width, label="Matryoshka SAE"
        )
        plt.bar(x + width / 2, comparison_df["Res-JB"], width, label="Res-JB SAE")

        plt.xlabel("Day Pairs")
        plt.ylabel("Co-occurrence")
        plt.title("Co-occurrence Comparison Between SAEs")
        plt.xticks(x, comparison_df["Day Pair"])
        plt.legend()
        plt.tight_layout()
        plt.savefig("day_cooccurrence_comparison.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    main()
