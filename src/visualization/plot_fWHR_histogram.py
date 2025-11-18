"""
plot_fWHR_histogram.py
---------------------------------------------------------
Plots a simple histogram for the facial width-to-height
ratio (fWHR) using the extracted feature CSV.

Author: Lizzie Qing
"""

import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_fWHR_histogram(
    csv_path: str,
    save_path: str = "results/fWHR_histogram.png"
):
    """
    Generates a histogram of fWHR values.

    Parameters
    ----------
    csv_path : str
        Path to the extracted features CSV.
    save_path : str
        Output file path for the saved plot.
    """
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.hist(
        df["fWHR"],
        bins=12,
        color="skyblue",
        edgecolor="black",
        alpha=0.85
    )

    plt.title("Face Width-to-Height Ratio (fWHR) Distribution")
    plt.xlabel("fWHR")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(save_path)
    plt.close()

    print(f"✓ fWHR histogram saved to: {save_path}")


if __name__ == "__main__":
    plot_fWHR_histogram(
        csv_path="animal_features_extracted.csv",
        save_path="results/fWHR_histogram.png"
    )
