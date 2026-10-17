"""Cluster map and dataset summary charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.engine import ART, load_processed

FIGURES = ART / "figures"


def plot() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = load_processed(ART / "problems.jsonl")
    coords = np.load(ART / "coords.npy")
    clusters = json.loads((ART / "clusters.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(11, 7))
    labels = frame["cluster"].to_numpy()
    cmap = plt.get_cmap("tab20")
    for cluster_id in sorted(set(labels.tolist())):
        mask = labels == cluster_id
        name = clusters[str(cluster_id)]["label"]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.75,
            color=cmap(int(cluster_id) % 20),
            label=f"{cluster_id}: {name[:42]} ({int(mask.sum())})",
        )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("CodeChef problems embedded with MiniLM, colored by K-means cluster")
    ax.legend(loc="best", fontsize=7, frameon=False, markerscale=1.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "clusters_pca.png", dpi=160)
    plt.close(fig)

    sizes = (
        frame.groupby(["cluster", "cluster_label"], as_index=False)
        .size()
        .sort_values("size", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sizes["cluster_label"], sizes["size"], color="#2a6f97")
    ax.set_xlabel("Problems")
    ax.set_title("Cluster sizes")
    fig.tight_layout()
    fig.savefig(FIGURES / "cluster_sizes.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(frame["difficulty"], bins=20, color="#bc4749", edgecolor="white")
    ax.set_xlabel("CodeChef difficulty rating")
    ax.set_ylabel("Problems")
    ax.set_title("Difficulty distribution of the indexed sample")
    fig.tight_layout()
    fig.savefig(FIGURES / "difficulty.png", dpi=160)
    plt.close(fig)

    tag_counts: dict[str, int] = {}
    for tags in frame["tags"]:
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    top = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)[:15]
    if top:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh([name for name, _ in top][::-1], [count for _, count in top][::-1], color="#6a994e")
        ax.set_xlabel("Problems")
        ax.set_title("Most common concept tags")
        fig.tight_layout()
        fig.savefig(FIGURES / "top_tags.png", dpi=160)
        plt.close(fig)
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    plot()
