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
from matryoshka_cooc.subgraph_analysis import (
    analyze_features_with_pca,
    analyze_subgraph,
)


def main():
    """Main function to run the script."""
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description="Generate PCA visualizations for a Matryoshka SAE model")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to the SAE checkpoint")
    parser.add_argument("--layer", type=int, default=8, help="Layer to analyze")
    parser.add_argument("--cooc_dir", type=str, required=True, help="Directory containing co-occurrence data")
    parser.add_argument("--output_dir", type=str, default="./matryoshka_pca_results", help="Directory to save results")
    parser.add_argument("--subgraph_id", type=int, help="ID of the subgraph to analyze (if not specified, will analyze all significant subgraphs)")
    parser.add_argument("--threshold", type=float, default=1.5, help="Activation threshold")
    parser.add_argument("--n_batches", type=int, default=10, help="Number of batches to process")
    parser.add_argument("--no_show_plots", action="store_true", help="Don't display plots")
    parser.add_argument("--no_save_data", action="store_true", help="Don't save data and plots")
    parser.add_argument("--include_special_tokens", action="store_true", help="Include special tokens in processing")
    parser.add_argument("--max_examples", type=int, default=500, help="Maximum number of examples to process")
    
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
            layer=args.layer,
        )
    else:
        # Analyze all significant subgraphs
        analyze_features_with_pca(
            results_dir=args.cooc_dir,
            sae_path=args.sae_path,
            output_dir=args.output_dir,
            layer=args.layer,
        )


if __name__ == "__main__":
    main()