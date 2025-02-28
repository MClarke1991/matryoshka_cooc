import os

import pandas as pd
import torch
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

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


def load_matryoshka_sae(checkpoint_path: str) -> tuple[GlobalBatchTopKMatryoshkaSAE, dict]:
    """
    Load a trained matryoshka SAE model from a checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Tuple of (SAE model, configuration)
    """
    device = set_device()
    print(f"Using device: {device}")

    # Configuration for GPT2-small (should match train_gpt2_sae.py)
    cfg = get_default_cfg()
    cfg["model_name"] = "gpt2-small"
    cfg["layer"] = 8
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
    print(f"Loaded Matryoshka SAE from {checkpoint_path}")

    return sae, cfg


def find_most_activating_tokens(
    model: HookedTransformer,
    sae: GlobalBatchTopKMatryoshkaSAE,
    batch_size: int = 4096*10,
    activation_threshold: float = 0.0,
) -> dict:
    """
    Find the most activating token for each SAE feature
    
    Args:
        model: The language model
        sae: The SAE model
        batch_size: Batch size for processing
        activation_threshold: Threshold for considering a feature activated
        
    Returns:
        Dictionary mapping feature indices to (token_id, token_string, activation_value)
    """
    device = sae.W_dec.device
    vocab_size = model.cfg.d_vocab
    feature_count = sae.W_dec.shape[0]
    
    # Dictionary to store results
    most_activating_tokens = {}
    max_activations = torch.zeros(feature_count, device=device)
    max_token_ids = torch.zeros(feature_count, dtype=torch.long, device=device)
    
    print(f"Finding most activating tokens for {feature_count} features...")
    
    # Process vocabulary in batches
    for start_idx in tqdm(range(0, vocab_size, batch_size), desc="Processing tokens"):
        end_idx = min(start_idx + batch_size, vocab_size)
        batch_token_ids = torch.arange(start_idx, end_idx, device=device)
        
        # Get token embeddings
        with torch.no_grad():
            # Forward pass through embedding layer
            token_embeddings = model.W_E[batch_token_ids]            
            # Encode with SAE
            feature_activations = sae.encode(token_embeddings)
            
            # Find tokens that maximize activation for each feature
            for token_idx, token_id in enumerate(batch_token_ids):
                token_activation = feature_activations[token_idx]
                
                # Update max activations
                new_max_mask = token_activation > max_activations
                max_activations[new_max_mask] = token_activation[new_max_mask]
                max_token_ids[new_max_mask] = token_id
    
    # Convert results to dictionary
    for feature_idx in range(feature_count):
        token_id = max_token_ids[feature_idx].item()
        activation = max_activations[feature_idx].item()
        token_str = model.tokenizer.decode([token_id])
        
        if activation > activation_threshold:
            most_activating_tokens[feature_idx] = (token_id, token_str, activation)
    
    return most_activating_tokens


def save_results_to_csv(results: dict, output_file: str) -> None:
    """
    Save the most activating tokens to a CSV file
    
    Args:
        results: Dictionary mapping feature indices to (token_id, token_string, activation_value)
        output_file: Path to output CSV file
    """
    data = []
    for feature_idx, (token_id, token_str, activation) in results.items():
        data.append({
            "feature_idx": feature_idx,
            "token_id": token_id,
            "token_str": token_str,
            "activation": activation
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main() -> None:
    # Configuration
    layer = 0
    checkpoint_path = f"checkpoints/gpt2-small_blocks.{layer}.hook_resid_pre_24576_global-matryoshka-topk_32_0.0003_final.pt"
    output_file = f"sae_most_activating_tokens_layer_{layer}.csv"
    activation_threshold = 1.5  # Minimum activation value to include in results
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    # Load model
    device = set_device()
    print(f"Loading GPT-2 small model to {device}...")
    model = HookedTransformer.from_pretrained("gpt2-small").to(device)
    
    # Load SAE
    print(f"Loading SAE from {checkpoint_path}...")
    sae, _ = load_matryoshka_sae(checkpoint_path)
    
    # Find most activating tokens
    results = find_most_activating_tokens(
        model=model,
        sae=sae,
        activation_threshold=activation_threshold
    )
    
    # Save results
    save_results_to_csv(results, output_file)
    
    print(f"Found most activating tokens for {len(results)} features")


if __name__ == "__main__":
    main()