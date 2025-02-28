import copy

import torch
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from matryoshka_cooc.activation_store import ActivationsStore
from matryoshka_cooc.config import get_default_cfg, post_init_cfg
from matryoshka_cooc.sae import GlobalBatchTopKMatryoshkaSAE
from matryoshka_cooc.training import train_sae_group_seperate_wandb


def train_matryoshka_sae_gpt2_layer0():
    # Set up configuration
    cfg = get_default_cfg()

    # Configuration for GPT2-small
    cfg["model_name"] = "gpt2-small"
    cfg["layer"] = 0  # Target layer 0
    cfg["site"] = "resid_pre"  # Pre-residual activations
    cfg["dataset_path"] = "Skylion007/openwebtext"
    cfg["aux_penalty"] = 1 / 32
    cfg["lr"] = 3e-4
    cfg["input_unit_norm"] = False
    cfg["wandb_project"] = "gpt2-small-matryoshka-layer0"
    cfg["l1_coeff"] = 0.0
    cfg["act_size"] = 768  # GPT2-small's hidden size

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    cfg["device"] = device

    # Matryoshka-specific configurations
    cfg["sae_type"] = "global-matryoshka-topk"
    cfg["dict_size"] = 768 * 32  # Total dictionary size
    cfg["top_k"] = 32
    cfg["bandwidth"] = 0.001
    cfg["group_sizes"] = [768, 768, 768 * 2, 768 * 4]

    cfg["num_tokens"] = 1e8  # Number of tokens to train on
    cfg["model_batch_size"] = 32
    cfg["checkpoint_freq"] = 5000

    # Use bfloat16 for better numerical stability if available
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        cfg["model_dtype"] = torch.bfloat16
    else:
        cfg["model_dtype"] = torch.float32

    # Update config with derived values
    cfg = post_init_cfg(cfg)

    print(f"Training on {cfg['device']} with {cfg['model_dtype']}")

    # Load model
    model = (
        HookedTransformer.from_pretrained_no_processing(cfg["model_name"])
        .to(cfg["model_dtype"])
        .to(cfg["device"])
    )

    # Set up activation store
    activations_store = ActivationsStore(model, cfg)

    # Create the Matryoshka SAE
    sae = GlobalBatchTopKMatryoshkaSAE(cfg)

    # Train the model
    train_sae_group_seperate_wandb([sae], activations_store, model, [cfg])

    # Save the final model
    import os

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(sae.state_dict(), f"checkpoints/{cfg['name']}_final.pt")

    print(f"Training complete. Model saved to checkpoints/{cfg['name']}_final.pt")
    return sae, cfg


if __name__ == "__main__":
    train_matryoshka_sae_gpt2_layer0()
