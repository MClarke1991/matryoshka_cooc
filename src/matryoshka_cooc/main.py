import copy

import torch
from transformer_lens import HookedTransformer

from matryoshka_cooc.activation_store import ActivationsStore
from matryoshka_cooc.config import get_default_cfg, post_init_cfg
from matryoshka_cooc.sae import BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE
from matryoshka_cooc.training import train_sae_group_seperate_wandb

cfg = get_default_cfg()
cfg["model_name"] = "gemma-2-2b"
cfg["layer"] = 12
cfg["site"] = "resid_pre"
cfg["dataset_path"] = "Skylion007/openwebtext"
cfg["aux_penalty"] = 1 / 32
cfg["lr"] = 3e-4
cfg["input_unit_norm"] = False
cfg["dict_size"] = 16384
cfg["wandb_project"] = "batch-topk-matryoshka"
cfg["l1_coeff"] = 0.0
cfg["act_size"] = 2304
cfg["device"] = "cuda"
cfg["bandwidth"] = 0.001
cfg["top_k_matryoshka"] = [10, 10, 10, 10, 10]
cfg["group_sizes"] = [
    # 16384 // 64,
    16384 // 32,
    16384 // 16,
    16384 // 8,
    16384 // 4,
    16384 // 2,
    16384,
]
cfg["num_tokens"] = 5e8
cfg["model_batch_size"] = 32
cfg["model_dtype"] = torch.bfloat16


# Train the BatchTopK SAEs
# dict_sizes = [2304, 2304 * 2, 2304 * 4, 2304 * 8, 2304 * 16]
# topks = [22, 25, 27, 29, 32]

# model = (
#     HookedTransformer.from_pretrained_no_processing(cfg["model_name"])
#     .to(cfg["model_dtype"])
#     .to(cfg["device"])
# )
# activations_store = ActivationsStore(model, cfg)
# saes = []
# cfgs = []

# for i, (dict_size, topk) in enumerate(zip(dict_sizes, topks)):
#     cfg = copy.deepcopy(cfg)
#     cfg["sae_type"] = "batch-topk"
#     cfg["dict_size"] = dict_size
#     cfg["top_k"] = topk

#     cfg = post_init_cfg(cfg)
#     sae = BatchTopKSAE(cfg)
#     saes.append(sae)
#     cfgs.append(cfg)

# Train the Matryoshka SAE
# dict_size = 2304 * 16
topk = 32
cfg = copy.deepcopy(cfg)
cfg["sae_type"] = "global-matryoshka-topk"
# cfg["dict_size"] = dict_size
cfg["top_k"] = topk
cfg["group_sizes"] = [
    dict_size // 16,
    dict_size // 16,
    dict_size // 8,
    dict_size // 4,
    dict_size // 2,
]

cfg = post_init_cfg(cfg)
sae = GlobalBatchTopKMatryoshkaSAE(cfg)
saes.append(sae)
cfgs.append(cfg)

train_sae_group_seperate_wandb(saes, activations_store, model, cfgs)
