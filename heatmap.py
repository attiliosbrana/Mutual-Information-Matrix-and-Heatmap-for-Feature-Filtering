import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gzip
import struct
from sklearn.metrics import normalized_mutual_info_score
from matplotlib.colors import LinearSegmentedColormap
import mplcatppuccin
from mplcatppuccin.colormaps import get_colormap_from_list
import seaborn as sns
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore", category=UserWarning)


def norm_comp_gain(x, y):
    """
    Computes the Normalized Compression Gain (NCG) between two sequences x and y.

    Steps:
    1. Quantize the sequences and convert to bytes for compression.
    2. Concatenate the byte sequences.
    3. Compress the individual and concatenated byte sequences using gzip.
    4. Calculate the actual compression gain.
    5. Define the best and worst case scenarios for compression.
    6. Normalize the gain by the difference between the worst and best case scenarios.
    7. Clip the normalized gain to [0, 1].
    8. Apply exponential scaling to accentuate the differences.
    9. Return the exponentially scaled normalized compression gain.
    """
    # x_quantized = x.astype(np.float16)
    # y_quantized = y.astype(np.float16)
    # x_bytes = x_quantized.tobytes()
    # y_bytes = y_quantized.tobytes()

    x_quantized = (x * 10**3).astype(np.int8)
    y_quantized = (y * 10**3).astype(np.int8)
    x_bytes = x_quantized.tobytes()
    y_bytes = y_quantized.tobytes()

    # x_bytes = x.tostring()
    # y_bytes = y.tostring()

    xy_bytes = x_bytes + y_bytes

    Cx = len(gzip.compress(x_bytes))
    Cy = len(gzip.compress(y_bytes))
    Cxy = len(gzip.compress(xy_bytes))

    gain = (Cx + Cy) - Cxy  # Actual gain
    worst_case = Cx + Cy  # Worst case scenario where there's no gain
    best_case = min(Cx, Cy)  # Best case scenario where gain is maximized

    # Avoid division by zero if worst_case equals best_case
    if worst_case == best_case:
        return 0.0

    normalized_gain = gain / (worst_case - best_case)  # Normalize the gain

    # Clip it to one or zero
    if normalized_gain > 1.0:
        normalized_gain = 1.0

    if normalized_gain < 0.0:
        normalized_gain = 0.0

    # Apply exponential scaling
    exp_scaled_gain = 1 - np.exp(
        -10 * normalized_gain
    )  # The factor 10 controls the rate of increase of the exponential function

    return exp_scaled_gain


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


def custom_colormap_ncg():
    return LinearSegmentedColormap.from_list(
        "ncg_colormap",
        [
            (0.0, "#1e1e2e40"),
            (1.0, "#FE640B"),
        ],
    )


def plot_combined_heatmap(
    df, target, lags=[0, 1], k=16, upper_triangle="nmi", lower_triangle="corr"
):
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

    def get_metric_function(metric):
        if metric == "corr":
            return lambda x, y: np.corrcoef(df[x], df[y])[0, 1]
        elif metric == "nmi":
            return lambda x, y: normalized_mutual_info_score(df[x], df[y])
        elif metric == "ncg":
            return lambda x, y: norm_comp_gain(df[x].to_numpy(), df[y].to_numpy())
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def compute_matrix(metric):
        metric_function = get_metric_function(metric)
        matrix = np.zeros((len(df.columns), len(df.columns)))
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                matrix[i, j] = metric_function(col1, col2)
        return matrix

    # Compute the upper triangle matrix to sort features based on the metric with respect to the target variable
    upper_triangle_matrix = compute_matrix(upper_triangle)
    target_index = df.columns.get_loc(target)
    upper_triangle_scores = {
        col: upper_triangle_matrix[target_index, i]
        for i, col in enumerate(df.columns)
        if col != target
    }
    sorted_columns_by_upper_metric = sorted(
        upper_triangle_scores.keys(),
        key=lambda x: upper_triangle_scores[x],
        reverse=True,
    )

    # Move the target variable and the sorted features to the beginning of the dataframe
    df = df[[target] + sorted_columns_by_upper_metric]

    # Keep only the top k features
    df = df.iloc[:, : k + 1]

    lower_triangle_matrix = compute_matrix(lower_triangle)
    upper_triangle_matrix = compute_matrix(upper_triangle)  # Recompute after sorting

    mask_lower = np.triu(np.ones_like(lower_triangle_matrix, dtype=bool))
    mask_upper = np.tril(np.ones_like(upper_triangle_matrix, dtype=bool))

    # Define a dictionary to map metric names to labels
    metric_labels = {
        "corr": "Correlation Coefficient",
        "nmi": "Normalized Mutual Information",
        "ncg": "Normalized Compression Gain",
    }

    # Look up the label for the upper triangle metric
    upper_triangle_label = metric_labels.get(upper_triangle)
    lower_triangle_label = metric_labels.get(lower_triangle)

    # Define a dictionary to map metric names to colormaps
    metric_colormaps = {
        "corr": custom_colormap_corr,
        "nmi": custom_colormap_nmi,
        "ncg": custom_colormap_ncg,
    }

    # Look up the colormap for the upper and lower triangle metrics
    upper_triangle_colormap = metric_colormaps.get(
        upper_triangle, custom_colormap_nmi
    )()
    lower_triangle_colormap = metric_colormaps.get(
        lower_triangle, custom_colormap_corr
    )()

    plt.figure(figsize=(14, 12))

    # Plot lower triangle
    ax_lower = sns.heatmap(
        lower_triangle_matrix,
        mask=mask_lower,
        cmap=lower_triangle_colormap,
        annot=True,
        fmt=".2f",
        linewidths=1,
        linecolor="white",
        square=True,
        annot_kws={"size": 12},
        cbar_kws={"label": lower_triangle_label},
        vmin=-1 if lower_triangle == "corr" else 0,  # Adjust vmin based on metric
        vmax=1,  # Setting maximum value for scaling
    )

    # Adjusting the transparency of annotations in the lower triangle
    for text in ax_lower.texts:
        text_value = float(text.get_text())
        text.set_alpha(0.15 + 0.85 * abs(text_value))

    # Plot upper triangle
    ax_upper = sns.heatmap(
        upper_triangle_matrix,
        mask=mask_upper,
        cmap=upper_triangle_colormap,
        annot=True,
        fmt=".2f",
        linewidths=1,
        linecolor="white",
        square=True,
        annot_kws={"size": 12},
        cbar_kws={"label": upper_triangle_label},
        vmin=0,  # Setting minimum value for scaling
        vmax=1,  # Setting maximum value for scaling
    )

    # Adjusting the transparency of annotations in the upper triangle
    for text in ax_upper.texts:
        text_value = float(text.get_text())
        text.set_alpha(0.15 + 0.85 * abs(text_value))

    # Adding a lighter border to the target variable's row and column
    light_red = "#d20f39"
    target_index = 0  # Target variable is now the first column
    ax_lower.add_patch(
        Rectangle(
            (0, target_index), len(df.columns), 1, fill=False, edgecolor=light_red, lw=2
        )
    )
    ax_lower.add_patch(
        Rectangle(
            (target_index, 0), 1, len(df.columns), fill=False, edgecolor=light_red, lw=2
        )
    )

    plt.xticks(np.arange(len(df.columns)) + 0.5, df.columns, rotation=90, fontsize=14)
    plt.yticks(np.arange(len(df.columns)) + 0.5, df.columns, rotation=0, fontsize=14)

    # Get the tick labels objects
    xticklabels = ax_lower.get_xticklabels()
    yticklabels = ax_lower.get_yticklabels()

    # Set the color of the target variable's label to red
    xticklabels[target_index].set_color("red")
    yticklabels[target_index].set_color("red")

    # Set the updated labels
    ax_lower.set_xticklabels(xticklabels)
    ax_lower.set_yticklabels(yticklabels)

    plt.title("Combined Heatmap", fontsize=20)

    plt.show()
