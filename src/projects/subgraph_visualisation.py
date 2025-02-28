# %% [markdown]
# # Analyzing Subgraph 516
# This notebook demonstrates how to analyze a specific subgraph from a trained Matryoshka SAE model.

# %% [markdown]
# First, import the required functions and libraries:

# %%
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from matryoshka_cooc.subgraph_analysis import analyze_subgraph

# Set paths and parameters
root_path = Path(__file__).parent.parent.parent
SAE_PATH = root_path / "checkpoints/gpt2-small_blocks.8.hook_resid_pre_24576_global-matryoshka-topk_32_0.0003_final.pt"
COOC_DIR = root_path / "graph_cooc_n_batches_500_layer_8/matryoshka"
SUBGRAPH_ID = 516
LAYER = 8

# %% [markdown]
# Run the analysis:

# %%
analysis = analyze_subgraph(
    sae_path=SAE_PATH,
    cooc_dir=COOC_DIR,
    subgraph_id=SUBGRAPH_ID,
    layer=LAYER,
    n_batches=10,
    max_examples=500
)

# %% [markdown]
# Now let's create some visualizations of the PCA results. First, let's create a function for token-based visualization:

# %%
def plot_token_pca(pca_df, subgraph_id, title="PCA Analysis by Token"):
    """Create an interactive scatter plot of PCA results colored by token"""
    # Get unique tokens for coloring
    unique_tokens = pca_df["tokens"].unique()
    
    # Use a good color scheme that will work with many tokens
    color_scale = px.colors.qualitative.Dark24
    color_map = {token: color_scale[i % len(color_scale)] 
                for i, token in enumerate(unique_tokens)}
    
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
                    showlegend=(i == 0),  # Only show legend for first subplot
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
    
    return fig

# Plot PCA results by token
token_fig = plot_token_pca(analysis["pca_df"], SUBGRAPH_ID)
token_fig.show()

# %% [markdown]
# Now let's create a visualization based on feature activations:

# %%
def plot_pca_by_feature_activation(pca_df, results, feature_list, subgraph_id):
    """Create PCA plots colored by top activated feature"""
    # Function to get the active feature for each data point
    def get_active_feature(idx):
        activations = results["all_graph_feature_acts"][idx].cpu().numpy()
        if np.all(activations == 0):
            return "None"
        active_idx = np.argmax(activations)
        return str(feature_list[active_idx])
    
    # Add column for the active feature
    pca_df = pca_df.copy()  # Create copy to avoid modifying original
    pca_df["active_feature"] = [get_active_feature(i) for i in range(len(pca_df))]
    
    # Create color map
    unique_features = pca_df["active_feature"].unique()
    color_scale = px.colors.qualitative.Dark24
    color_map = {feature: color_scale[i % len(color_scale)] 
                for i, feature in enumerate(unique_features)}
    
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
                    showlegend=(i == 0),
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
    
    return fig

# Plot PCA results by feature activation
feature_fig = plot_pca_by_feature_activation(
    analysis["pca_df"], 
    analysis["results"], 
    analysis["subgraph_nodes"], 
    SUBGRAPH_ID
)
feature_fig.show()

# %% [markdown]
# Let's also look at the feature activation patterns:

# %%
def plot_feature_activations(results, feature_list, subgraph_id, max_examples=100):
    """Create heatmap of feature activations across examples"""
    # Get feature activations
    feature_activations = results["all_graph_feature_acts"].cpu().numpy()
    
    # Limit to first max_examples examples for better visualization
    feature_activations = feature_activations[:max_examples]
    
    # Create heatmap
    fig = px.imshow(
        feature_activations,
        labels=dict(x="Feature Index", y="Example", color="Activation"),
        x=[f"Feature {i}" for i in feature_list[:feature_activations.shape[1]]],
        title=f"Feature Activations for Subgraph {subgraph_id} (First {max_examples} Examples)",
        color_continuous_scale="Viridis",
    )
    
    return fig

# Plot feature activations
activation_fig = plot_feature_activations(
    analysis["results"], 
    analysis["subgraph_nodes"], 
    SUBGRAPH_ID
)
activation_fig.show()

# %% [markdown]
# We can also create a 3D visualization of the PCA results:

# %%
def plot_3d_pca(pca_df, subgraph_id, color_by="token"):
    """Create a 3D scatter plot visualization of PCA results"""
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
        # Add active feature column if not present
        if "active_feature" not in pca_df.columns:
            pca_df = pca_df.copy()
            def get_active_feature(idx):
                activations = results["all_graph_feature_acts"][idx].cpu().numpy()
                return str(feature_list[np.argmax(activations)])
            pca_df["active_feature"] = [get_active_feature(i) for i in range(len(pca_df))]
            
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
    return fig

# Create 3D visualizations
pca_3d_token = plot_3d_pca(analysis["pca_df"], SUBGRAPH_ID, color_by="token")
pca_3d_token.show()

# %% [markdown]
# Finally, let's analyze the variance explained by each principal component:

# %%
def plot_explained_variance(pca):
    """Plot the explained variance ratio for each principal component"""
    exp_var_ratio = pca.explained_variance_ratio_
    cum_exp_var_ratio = np.cumsum(exp_var_ratio)
    
    fig = go.Figure()
    
    # Bar plot for individual explained variance
    fig.add_trace(go.Bar(
        x=[f"PC{i+1}" for i in range(len(exp_var_ratio))],
        y=exp_var_ratio * 100,
        name="Individual"
    ))
    
    # Line plot for cumulative explained variance
    fig.add_trace(go.Scatter(
        x=[f"PC{i+1}" for i in range(len(cum_exp_var_ratio))],
        y=cum_exp_var_ratio * 100,
        name="Cumulative",
        yaxis="y2"
    ))
    
    # Update layout
    fig.update_layout(
        title="Explained Variance Ratio by Principal Component",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance Ratio (%)",
        yaxis2=dict(
            title="Cumulative Explained Variance Ratio (%)",
            overlaying="y",
            side="right"
        ),
        height=500
    )
    
    return fig

# Plot explained variance
variance_fig = plot_explained_variance(analysis["pca"])
variance_fig.show()