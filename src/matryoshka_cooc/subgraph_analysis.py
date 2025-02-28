import os
from os.path import join as pj
from typing import Any, dict

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer


def set_device() -> str:
    """Set device based on availability"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_matryoshka_sae(checkpoint_path: str, layer: int, act_size: int = 768) -> tuple[Any, dict]:
    """Load a trained matryoshka SAE model and its config"""
    from matryoshka_cooc.config import get_default_cfg, post_init_cfg
    from matryoshka_cooc.sae import GlobalBatchTopKMatryoshkaSAE
    
    device = set_device()
    
    # Configuration for GPT2-small
    cfg = get_default_cfg()
    cfg["model_name"] = "gpt2-small"
    cfg["layer"] = layer
    cfg["site"] = "resid_pre"
    cfg["act_size"] = act_size
    cfg["device"] = device

    # Matryoshka-specific configurations
    cfg["sae_type"] = "global-matryoshka-topk"
    cfg["dict_size"] = act_size * 32  # Total dictionary size
    cfg["top_k"] = 32
    cfg["group_sizes"] = [act_size, act_size, act_size * 2, act_size * 4]

    # Update config with derived values
    cfg = post_init_cfg(cfg)

    # Create model with the same architecture
    sae = GlobalBatchTopKMatryoshkaSAE(cfg)

    # Load trained weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    sae.load_state_dict(state_dict)
    
    return sae, cfg

def load_node_info(cooc_dir: str, sae_name: str, threshold: float) -> pd.DataFrame:
    """Load node information for the subgraph analysis"""
    threshold_str = str(threshold).replace(".", "_")
    
    # Try direct path first
    node_info_file = pj(cooc_dir, "dataframes", f"{sae_name}_node_info_{threshold_str}.csv")
    if os.path.exists(node_info_file):
        return pd.read_csv(node_info_file)
    
    # Try alternate path structure
    alt_node_info_file = pj(cooc_dir, "graph_results", sae_name, "dataframes", 
                          f"{sae_name}_node_info_{threshold_str}.csv")
    if os.path.exists(alt_node_info_file):
        return pd.read_csv(alt_node_info_file)
    
    raise FileNotFoundError(f"Could not find node info file in {cooc_dir}")

def list_flatten(nested_list):
    """Flattens a nested list into a single-level list"""
    return [x for y in nested_list for x in y]

def make_token_df(tokens, model, len_prefix=10, len_suffix=10) -> pd.DataFrame:
    """Create a DataFrame containing token information and context"""
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

def get_special_tokens(model) -> set:
    """Get a set of special tokens for the model"""
    special_tokens = set()
    special_tokens.add(model.tokenizer.bos_token_id)
    special_tokens.add(model.tokenizer.eos_token_id)
    special_tokens.add(model.tokenizer.pad_token_id)
    return {token for token in special_tokens if token is not None}

def process_examples(
    activation_store,
    model,
    sae,
    feature_list: list[int],
    n_batches: int,
    remove_special_tokens: bool = True,
    device: str = "cpu",
    max_examples: int = 500,
    layer: int = 8,
) -> dict:
    """Process examples from the activation store using the given model and SAE"""
    examples_found = 0
    all_fired_tokens = []
    all_graph_feature_acts = []
    all_max_feature_info = []
    all_reconstructions = []
    all_token_dfs = []

    feature_list_tensor = torch.tensor(feature_list, device=sae.W_dec.device)

    pbar = tqdm(range(n_batches), leave=True)
    for _ in pbar:
        # Get a batch of tokens from the activation store
        tokens = activation_store.get_batch_tokens()

        # Create a DataFrame containing token information
        tokens_df = make_token_df(tokens, model)

        # Flatten the tokens tensor for easier processing
        flat_tokens = tokens.flatten()

        # Run the model and get activations
        hook_point = sae.config["hook_point"]
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_point],
                stop_at_layer=layer + 1,
            )
        sae_in = cache[hook_point]
        
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

        # Get max feature info
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

        # Reconstruct using features in feature_list
        reconstruction = (
            feature_acts[fired_mask][:, feature_list] @ sae.W_dec[feature_list]
        )

        # Store results
        all_token_dfs.append(
            tokens_df.iloc[fired_mask.cpu().nonzero().flatten().numpy()]
        )
        all_graph_feature_acts.append(feature_acts[fired_mask][:, feature_list])
        all_max_feature_info.append(max_feature_info)
        all_fired_tokens.append(fired_tokens)
        all_reconstructions.append(reconstruction)

        examples_found += len(fired_tokens)
        pbar.set_description(f"Examples found: {examples_found}")

        if max_examples is not None and examples_found >= max_examples:
            break

    # Process results
    all_token_dfs = pd.concat(all_token_dfs) if all_token_dfs else pd.DataFrame()
    all_fired_tokens = list_flatten(all_fired_tokens)
    all_reconstructions = torch.cat(all_reconstructions) if all_reconstructions else torch.tensor([])
    all_graph_feature_acts = torch.cat(all_graph_feature_acts) if all_graph_feature_acts else torch.tensor([])
    all_max_feature_info = torch.cat(all_max_feature_info) if all_max_feature_info else torch.tensor([])

    # Limit to max_examples if needed
    if max_examples is not None and len(all_fired_tokens) > max_examples:
        all_token_dfs = all_token_dfs.iloc[:max_examples]
        all_fired_tokens = all_fired_tokens[:max_examples]
        all_reconstructions = all_reconstructions[:max_examples]
        all_graph_feature_acts = all_graph_feature_acts[:max_examples]
        all_max_feature_info = all_max_feature_info[:max_examples]

    return {
        "all_token_dfs": all_token_dfs,
        "all_fired_tokens": all_fired_tokens,
        "all_reconstructions": all_reconstructions,
        "all_graph_feature_acts": all_graph_feature_acts,
        "all_examples_found": examples_found,
        "all_max_feature_info": all_max_feature_info,
    }

def perform_pca(results: dict, n_components: int = 3) -> tuple[pd.DataFrame | None, PCA | None]:
    """Perform PCA on the reconstructions and return results DataFrame"""
    reconstructions = results["all_reconstructions"]
    n_samples, n_features = reconstructions.detach().cpu().numpy().shape
    max_components = min(n_samples, n_features)

    if n_components > max_components:
        print(f"Warning: Requested {n_components} components but only {max_components} possible")
        return None, None

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_embedding = pca.fit_transform(reconstructions.cpu().numpy())

    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        pca_embedding, 
        columns=pd.Index([f"PC{i+1}" for i in range(n_components)])
    )

    # Add tokens and context information
    pca_df["tokens"] = results["all_fired_tokens"]
    pca_df["context"] = results["all_token_dfs"].context.values
    pca_df["point_id"] = range(len(pca_df))

    return pca_df, pca

def analyze_subgraph(
    sae_path: str,
    cooc_dir: str,
    subgraph_id: int,
    layer: int = 8,
    activation_threshold: float = 1.5,
    n_batches: int = 10,
    remove_special_tokens: bool = True,
    max_examples: int = 500,
) -> dict:
    """Analyze a specific subgraph and return the analysis results"""
    device = set_device()
    
    # Load SAE model
    sae, cfg = load_matryoshka_sae(sae_path, layer)
    
    # Load model
    model = HookedTransformer.from_pretrained(cfg["model_name"], device=device)
    
    # Load node information
    node_df = load_node_info(cooc_dir, "matryoshka", activation_threshold)
    
    # Filter for the specific subgraph
    subgraph_nodes = node_df.query(f"subgraph_id == {subgraph_id}")["node_id"].tolist()
    
    if not subgraph_nodes:
        raise ValueError(f"No nodes found for subgraph {subgraph_id}")
    
    # Set up activation store
    from matryoshka_cooc.activation_store import ActivationsStore
    activation_store = ActivationsStore(model, cfg)
    
    # Process examples
    results = process_examples(
        activation_store,
        model,
        sae,
        subgraph_nodes,
        n_batches,
        remove_special_tokens=remove_special_tokens,
        device=device,
        max_examples=max_examples,
        layer=layer,
    )
    
    # Perform PCA
    pca_df, pca = perform_pca(results)
    
    return {
        "results": results,
        "pca_df": pca_df,
        "pca": pca,
        "sae": sae,
        "model": model,
        "subgraph_nodes": subgraph_nodes,
    }