"""
plot_training_log.py
---------------------------------------------------------
Plots YOLOv8 training curves (Loss & mAP) using results.csv.
Includes optional smoothing with exponential moving average.

Author: Lizzie Qing
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# -------------------------------------------------------
# Helper: smooth curves with exponential moving average
# -------------------------------------------------------
def smooth_curve(values, smoothing_factor=0.8):
    """
    Smooths a sequence using exponential moving average.

    Parameters
    ----------
    values : list or array
        Raw time-series values.
    smoothing_factor : float
        Weight for smoothing (0–1). Higher → smoother.

    Returns
    -------
    np.ndarray
        Smoothed values.
    """
    smoothed = []
    last = values[0]
    for v in values:
        last = last * smoothing_factor + (1 - smoothing_factor) * v
        smoothed.append(last)
    return np.array(smoothed)


# -------------------------------------------------------
# Main plotting function
# -------------------------------------------------------
def plot_training_log(
    csv_path: str,
    save_dir: str = "results/training_plots"
):
    """
    Plots smoothed training loss and validation mAP curves
    from YOLOv8 results.csv.

    Parameters
    ----------
    csv_path : str
        Path to YOLO 'results.csv'.
    save_dir : str
        Directory where plots will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # ----------------------------------
    # Plot training & validation loss
    # ----------------------------------
    loss_cols = [
        ("train/box_loss", "Train Box Loss"),
        ("train/cls_loss", "Train Class Loss"),
        ("train/dfl_loss", "Train DFL Loss"),
        ("val/box_loss", "Val Box Loss"),
        ("val/cls_loss", "Val Class Loss"),
        ("val/dfl_loss", "Val DFL Loss"),
    ]

    plt.figure(figsize=(12, 6))
    for col, label in loss_cols:
        if col in df.columns:
            plt.plot(
                df["epoch"],
                smooth_curve(df[col]),
                label=label
            )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("YOLOv8 Training & Validation Loss Curves (Smoothed)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curves.png"))
    plt.close()

    # ----------------------------------
    # Plot mAP curves
    # ----------------------------------
    plt.figure(figsize=(12, 6))
    if "metrics/mAP50(B)" in df.columns:
        plt.plot(
            df["epoch"],
            smooth_curve(df["metrics/mAP50(B)"]),
            label="mAP@0.5"
        )
    if "metrics/mAP50-95(B)" in df.columns:
        plt.plot(
            df["epoch"],
            smooth_curve(df["metrics/mAP50-95(B)"]),
            label="mAP@0.5:0.95"
        )

    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("YOLOv8 Validation mAP Curves (Smoothed)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mAP_curves.png"))
    plt.close()

    print(f"✓ Training plots saved to: {save_dir}/")


# -------------------------------------------------------
# Script entry point
# -------------------------------------------------------
if __name__ == "__main__":
    plot_training_log(
        csv_path="runs/detect_yolov8s_finetune_v2/results.csv",
        save_dir="results/training_plots"
    )
