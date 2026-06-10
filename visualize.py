"""
Matplotlib helpers for visualizing the network.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def plot_network_architecture(layer_sizes: list[int], activations: list[np.ndarray] | None = None):
    """Draw the network as circles + connecting lines.

    If `activations` is provided (one array per non-input layer, including output),
    neuron color brightness reflects activation strength.
    """
    # Cap displayed neurons per layer so the diagram stays readable
    display_cap = 16
    displayed = [min(s, display_cap) for s in layer_sizes]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")

    x_positions = np.linspace(0, 1, len(layer_sizes))
    layer_coords = []

    for li, (n_show, n_real, x) in enumerate(zip(displayed, layer_sizes, x_positions)):
        ys = np.linspace(0.1, 0.9, n_show) if n_show > 1 else np.array([0.5])
        layer_coords.append(list(zip([x] * n_show, ys)))

        # Activation values for coloring (skip input layer)
        act_vec = None
        if activations is not None and li > 0 and li - 1 < len(activations):
            a = activations[li - 1].flatten()
            if a.size:
                a_norm = (a - a.min()) / (np.ptp(a) + 1e-9)
                act_vec = a_norm[:n_show]

        for i, (cx, cy) in enumerate(zip([x] * n_show, ys)):
            intensity = float(act_vec[i]) if act_vec is not None else 0.35
            color = (0.2 + 0.8 * intensity, 0.5 + 0.3 * intensity, 1.0)
            ax.add_patch(Circle((cx, cy), 0.018, color=color, ec="white", lw=0.6, zorder=3))

        # Label
        label = f"Input\n({n_real})" if li == 0 else (
            f"Output\n({n_real})" if li == len(layer_sizes) - 1 else f"Hidden {li}\n({n_real})"
        )
        ax.text(x, 0.02, label, ha="center", color="white", fontsize=9)
        if n_real > n_show:
            ax.text(x, 0.95, f"showing {n_show}/{n_real}", ha="center",
                    color="#888", fontsize=7, style="italic")

    # Connections between layers (subset, for clarity)
    for li in range(len(layer_coords) - 1):
        for (x1, y1) in layer_coords[li]:
            for (x2, y2) in layer_coords[li + 1]:
                ax.plot([x1, x2], [y1, y2], color="#3a3f55", lw=0.4, zorder=1)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    return fig


def plot_history(history) -> "plt.Figure":
    """Plot training and validation loss/accuracy curves from Keras history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
    fig.patch.set_facecolor("#0e1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#555")

    epochs = range(1, len(history.history["loss"]) + 1)
    ax1.plot(epochs, history.history["loss"], "o-", color="#ff6b6b", label="train")
    ax1.plot(epochs, history.history["val_loss"], "o--", color="#ffd166", label="val")
    ax1.set_title("Loss (lower = better)", color="white")
    ax1.set_xlabel("Epoch", color="white")
    ax1.legend(facecolor="#0e1117", edgecolor="#555", labelcolor="white")

    ax2.plot(epochs, history.history["accuracy"], "o-", color="#06d6a0", label="train")
    ax2.plot(epochs, history.history["val_accuracy"], "o--", color="#118ab2", label="val")
    ax2.set_title("Accuracy (higher = better)", color="white")
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylim(0, 1.02)
    ax2.legend(facecolor="#0e1117", edgecolor="#555", labelcolor="white")

    fig.tight_layout()
    return fig


def plot_prediction_bars(probs: np.ndarray):
    """Plot the output probabilities as a bar chart, highlighting the top guess."""
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#555")

    classes = list(range(10))
    colors = ["#06d6a0" if i == int(np.argmax(probs)) else "#3a86ff" for i in classes]
    ax.bar(classes, probs, color=colors)
    ax.set_xticks(classes)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Digit", color="white")
    ax.set_ylabel("Confidence", color="white")
    ax.set_title("What the network thinks", color="white")
    fig.tight_layout()
    return fig
