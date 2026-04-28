import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *

channel_names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def build_channel_stat_dataframe(segment_manifest_df, stat_type="mean"):
    """
    Build a compact dataframe containing one column per channel statistic.
    stat_type options:
        - "mean"
        - "std"
        - "rms"
    """
    valid_stats = {"mean", "std", "rms"}
    if stat_type not in valid_stats:
        raise ValueError(f"stat_type must be one of {valid_stats}")

    cols = [f"{ch}_{stat_type}" for ch in channel_names]

    missing = [col for col in cols if col not in segment_manifest_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = segment_manifest_df[cols].copy()
    df.columns = channel_names
    return df


def compute_channel_correlation(segment_manifest_df, stat_type="mean", group=None):
    """
    Compute channel correlation matrix from segment-level features.
    Optionally filter by group: Static / Dynamic / Transition
    """
    df = segment_manifest_df.copy()

    if group is not None:
        df = df[df["group"] == group].copy()

    stat_df = build_channel_stat_dataframe(df, stat_type=stat_type)
    corr_df = stat_df.corr() # Performs the pearson correlation

    return corr_df


def plot_channel_correlation_heatmap(corr_df, title="", save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_df.values, interpolation="nearest", aspect="auto")

    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.index)

    ax.set_title(title)

    # Annotate values
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            ax.text(
                j, i,
                f"{corr_df.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_grouped_channel_correlation(segment_manifest_df, stat_type="mean", save_dir=None):
    """
    Plot 3 compact heatmaps in one figure:
    Static, Dynamic, Transition
    """
    groups = ["Static", "Dynamic", "Transition"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), squeeze=False)
    axes = axes.flatten()

    ims = []

    for ax, group in zip(axes, groups):
        corr_df = compute_channel_correlation(
            segment_manifest_df=segment_manifest_df,
            stat_type=stat_type,
            group=group,
        )

        im = ax.imshow(corr_df.values, interpolation="nearest", aspect="auto", vmin=-1, vmax=1)
        ims.append(im)

        ax.set_xticks(np.arange(len(corr_df.columns)))
        ax.set_yticks(np.arange(len(corr_df.index)))
        ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr_df.index, fontsize=8)
        ax.set_title(f"{group} ({stat_type.upper()})")

        for i in range(corr_df.shape[0]):
            for j in range(corr_df.shape[1]):
                ax.text(
                    j, i,
                    f"{corr_df.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7
                )

    cbar = fig.colorbar(ims[0], ax=axes.tolist(), shrink=0.85)
    cbar.set_label("Correlation")

    fig.suptitle(f"Channel Correlation by Activity Group ({stat_type.upper()})", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"grouped_channel_correlation_{stat_type}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def print_channel_correlation_summary(corr_df, label="Overall"):
    print("\n" + "=" * 80)
    print(f"CHANNEL CORRELATION SUMMARY - {label}")
    print("=" * 80)
    print(corr_df.round(4))