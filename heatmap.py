import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from matplotlib.colors import LinearSegmentedColormap
import mplcatppuccin
from mplcatppuccin.colormaps import get_colormap_from_list
import seaborn as sns
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore", category=UserWarning)


def set_style():
    try:
        plt.style.use("mocha")
    except:
        plt.style.use("dark_background")


def custom_colormap_corr():
    return LinearSegmentedColormap.from_list(
        "corr_colormap",
        [
            (0.0, "#f38ba8"),
            (0.5, "#1e1e2e40"),  # Making the center transparent
            (1.0, "#a6e3a1"),
        ],
    )


def custom_colormap_nmi():
    return LinearSegmentedColormap.from_list(
        "nmi_colormap",
        [
            (0.0, "#1e1e2e40"),
            (1.0, "#89dceb"),
        ],
    )


def plot_combined_heatmap(df, target, lags=[0, 1], k=16):
    set_style()

    # Identify and Encode Categorical Variables
    for col in df.columns:
        if df[col].dtype.name == "object" or df[col].dtype.name == "category":
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
            df[col + "_numerical"] = pd.factorize(df[col])[0]
            df.drop(col, axis=1, inplace=True)

    # Creating lagged columns for continuous variables
    for col in df.columns:
        if df[col].dtype.name != "object" and df[col].dtype.name != "category":
            for lag in lags[1:]:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Drop rows with NaN values created by the lagging process
    df.dropna(inplace=True)

    # Calculate NMI scores between all features and the target
    nmi_scores = {
        col: normalized_mutual_info_score(df[col], df[target])
        for col in df.columns
        if col != target
    }

    # Sort features by NMI scores in descending order
    sorted_columns_by_nmi = sorted(
        nmi_scores.keys(), key=lambda x: nmi_scores[x], reverse=True
    )

    # Move the target variable and the sorted features to the beginning of the dataframe
    df = df[[target] + sorted_columns_by_nmi]

    # Find index of target variable
    target_index = df.columns.get_loc(target)

    # Keep only the top k features
    df = df.iloc[:, : k + 1]

    # Get the correlation matrix and NMI matrix
    corr_matrix = df.corr()
    nmi_matrix = np.zeros((len(df.columns), len(df.columns)))
    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            nmi_matrix[i, j] = normalized_mutual_info_score(df[col1], df[col2])

    # Create a new matrix where one half is the correlation matrix and the other half is the NMI matrix
    combined_matrix = np.zeros((len(df.columns), len(df.columns)))
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            if i > j:
                combined_matrix[i, j] = corr_matrix.iloc[i, j]
            else:
                combined_matrix[i, j] = nmi_matrix[i, j]

    mask_lower = np.triu(np.ones_like(combined_matrix, dtype=bool))
    mask_upper = np.tril(np.ones_like(combined_matrix, dtype=bool))

    plt.figure(figsize=(14, 12))

    # Plot lower triangle
    ax_lower = sns.heatmap(
        combined_matrix,
        mask=mask_lower,
        cmap=custom_colormap_corr(),
        annot=True,
        fmt=".2f",
        linewidths=1,
        linecolor="white",
        square=True,
        annot_kws={"size": 12},
        cbar_kws={"label": "Correlation Coefficient"},
        vmin=-1,  # Setting minimum value for scaling
        vmax=1,  # Setting maximum value for scaling
    )

    # Adjusting the transparency of annotations in the lower triangle
    for text in ax_lower.texts:
        text_value = float(text.get_text())
        text.set_alpha(0.15 + 0.85 * abs(text_value))

    # Plot upper triangle
    ax_upper = sns.heatmap(
        combined_matrix,
        mask=mask_upper,
        cmap=custom_colormap_nmi(),
        annot=True,
        fmt=".2f",
        linewidths=1,
        linecolor="white",
        square=True,
        annot_kws={"size": 12},
        cbar_kws={"label": "Normalized Mutual Information"},
        vmin=0,  # Setting minimum value for scaling
        vmax=1,  # Setting maximum value for scaling
    )

    # Adjusting the transparency of annotations in the upper triangle
    for text in ax_upper.texts:
        text_value = float(text.get_text())
        text.set_alpha(0.15 + 0.85 * abs(text_value))

    # Adding a lighter border to the target variable's row and column
    light_red = "#d20f39"
    ax_lower.add_patch(
        Rectangle(
            (0, target_index), len(df.columns), 1, fill=False, edgecolor=light_red, lw=2
        )
    )
    ax_lower.add_patch(
        Rectangle(
            (0, target_index - 1),
            len(df.columns),
            1,
            fill=False,
            edgecolor=light_red,
            lw=2,
        )
    )
    ax_lower.add_patch(
        Rectangle(
            (target_index, 0), 1, len(df.columns), fill=False, edgecolor=light_red, lw=2
        )
    )
    ax_lower.add_patch(
        Rectangle(
            (target_index - 1, 0),
            1,
            len(df.columns),
            fill=False,
            edgecolor=light_red,
            lw=2,
        )
    )

    plt.xticks(np.arange(len(df.columns)) + 0.5, df.columns, rotation=90, fontsize=14)
    plt.yticks(np.arange(len(df.columns)) + 0.5, df.columns, rotation=0, fontsize=14)

    # Find the index of the target variable in the column list
    target_index = list(df.columns).index(target)

    # Get the tick labels objects
    xticklabels = ax_lower.get_xticklabels()
    yticklabels = ax_lower.get_yticklabels()

    # Set the color of the target variable's label to red
    xticklabels[target_index].set_color("red")
    yticklabels[target_index].set_color("red")

    # Set the updated labels
    ax_lower.set_xticklabels(xticklabels)
    ax_lower.set_yticklabels(yticklabels)

    plt.title("Combined Correlation and NMI Heatmap", fontsize=20)

    plt.show()
