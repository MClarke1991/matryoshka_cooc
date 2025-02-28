# Matryoshka SAE latents co-occur strongly but remain independently interpretable

Code to investigate co-occurrence in Matryoshka SAEs. 

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage dependencies. 

## Usage

`src/projects/1_train_gpt2_sae.py` contains code to train a Matryoshka SAE on GPT2. 

`src/projects/2_get_cooc_mat_normal.py` contains code to get the co-occurrence matrix for a normal SAE and Matryoshka SAE. 

`src/projects/3_generate_cooc_tables.py` contains code to generate the co-occurrence tables for a Matryoshka SAE. 

`src/projects/4_summary_stats.ipynb` contains code to analyse the co-occurrence rates. 

`src/projects/5_pca.py` contains code to perform PCA on a specific co-occurrence matrix. 

`src/projects/6_sae_vis_basic.ipynb` contains code to visualise the SAE latents for a representative example. 

Representative data are contained in the zip file, unpack in root to use. 

### PCA Visualization Parameters

The PCA visualization script (`src/projects/5_pca.py`) accepts the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sae_path` | str | Required | Path to the SAE checkpoint |
| `--layer` | int | 8 | Layer to analyze |
| `--cooc_dir` | str | Required | Directory containing co-occurrence data |
| `--output_dir` | str | ./matryoshka_pca_results | Directory to save results |
| `--subgraph_id` | int | None | ID of the subgraph to analyze (if not specified, will analyze all significant subgraphs) |
| `--threshold` | float | 1.5 | Activation threshold |
| `--n_batches` | int | 10 | Number of batches to process |
| `--no_show_plots` | flag | False | Don't display plots |
| `--no_save_data` | flag | False | Don't save data and plots |
| `--include_special_tokens` | flag | False | Include special tokens in processing |
| `--max_examples` | int | 500 | Maximum number of examples to process |



## Acknowledgements

This inherits code from: https://github.com/bartbussmann/matryoshka_sae, thanks to the authors!