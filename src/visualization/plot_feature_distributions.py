"""
plot_feature_distributions.py
---------------------------------------------------------
Visualizes the distribution of extracted animal facial
features using KDE plots, grouped by animal type.

Author: Lizzie Qing
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------------------------------
# Helper: infer animal species from filename prefix
# -------------------------------------------------------
def infer_animal_type(filename: str) -> str:
    """
    Infers an animal category based on filename prefix.

    Expected patterns:
    - cat_xxx.png  → cat
    - dog_xxx.jpg  → dog
    - fox_xxx.png  → fox
    - hamster_xxx.jpg → hamster
    """
    name = filename.lower()
    if name.startswith("cat"):
        return "cat"
    elif name.startswith("dog"):
        return "dog"
    elif name.startswith("fox"):
        return "fox"
    elif name.startswith("hamster"):
        return "hamster"
    return "other"


# -------------------------------------------------------
# Main visualization function
# -------------------------------------------------------
def plot_feature_distributions(
    csv_path: str,
    save_dir: str = "feature_distribution_plots"
):
    """
    Generates KDE distribution plots for selected features,
    grouped by animal type.

    Parameters
    ----------
    csv_path : str
        Path to the extracted features CSV.
    save_dir : str
        Folder where plots will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load feature table
    df = pd.read_csv(csv_path)
    df["animal_type"] = df["filename"].apply(infer_animal_type)

    # Features to visualize
    features_to_plot = ["EFR", "ESI", "avg_ear_uprightness_deg", "fWHR"]

    for feature in features_to_plot:
        plt.figure(figsize=(10, 6))

        # KDE curves by type
        sns.kdeplot(
            data=df,
            x=feature,
            hue="animal_type",
            fill=True,
            common_norm=False,
            alpha=0.5,
            linewidth=2
        )

        # Stats
        mean_val = df[feature].mean()
        std_val = df[feature].std()

        # Mean line
        plt.axvline(mean_val, color="black", linestyle="--",
                    label=f"Mean: {mean_val:.3f}")

        # ±1 std interval
        plt.axvline(mean_val + std_val, color="gray", linestyle=":",
                    label=f"+1σ: {mean_val + std_val:.3f}")
        plt.axvline(mean_val - std_val, color="gray", linestyle=":",
                    label=f"-1σ: {mean_val - std_val:.3f}")

        plt.title(f"{feature} Distribution", fontsize=16)
        plt.xlabel(feature, fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        save_path = os.path.join(save_dir, f"{feature}_distribution.png")
        plt.savefig(save_path)
        plt.close()

    print(f"✓ Feature distribution plots saved to: {save_dir}/")


# -------------------------------------------------------
# Script entry point
# -------------------------------------------------------
if __name__ == "__main__":
    plot_feature_distributions(
        csv_path="animal_features_extracted.csv",
        save_dir="feature_distribution_plots"
    )
