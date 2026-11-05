"""Model-level SVG comparisons for assessment summaries."""

import numpy as np
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure

from src.models.ehr_metrics import ENDPOINTS


def interval_plot(results, metrics, path, *, title):
    keys = list(results)
    figure = Figure(figsize=(12, max(8, 2.3 + len(keys) * 1.15)), layout="constrained")
    figure.suptitle(title)
    axes = figure.subplots(3, 2).ravel()
    for axis, (key, label) in zip(axes, metrics, strict=True):
        axis.set_title(label)
        axis.set_yticks(range(len(keys)), keys)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
        if key.endswith("_relative_pct"):
            axis.axvline(0, color="gray", linestyle="--", linewidth=1)
        else:
            axis.set_xlim(-0.025, 1.025)
        for i, name in enumerate(keys):
            estimate = results[name][key]
            if estimate["lower"] is not None:
                axis.hlines(i, estimate["lower"], estimate["upper"], color="C0")
                axis.scatter([estimate["lower"], estimate["upper"]], [i, i], color="C0", marker="|")
            if estimate["point"] is not None:
                axis.scatter(estimate["point"], i, color="C0")
    FigureCanvasSVG(figure).print_svg(path)


def confusion_plot(results, path, *, title):
    figure = Figure(figsize=(2.55 * len(results), 5.7), layout="constrained")
    figure.suptitle(title)
    axes = figure.subplots(len(ENDPOINTS), len(results), squeeze=False)
    for row, endpoint in enumerate(ENDPOINTS):
        for column, (key, endpoints) in enumerate(results.items()):
            axis = axes[row, column]
            counts = endpoints[endpoint]["confusion_matrix"]
            matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
            totals = matrix.sum(axis=1, keepdims=True)
            percentages = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)
            axis.imshow(percentages, vmin=0, vmax=1, cmap="Blues")
            for i, j in np.ndindex(matrix.shape):
                fraction = percentages[i, j]
                axis.text(j, i, f"{matrix[i, j]}\n{fraction:.1%}", ha="center", va="center",
                          color="white" if fraction > 0.55 else "black")
            axis.set_xticks([0, 1], ["Negative", "Positive"])
            axis.set_yticks([0, 1], ["Negative", "Positive"])
            axis.set_title(key if row == 0 else "")
            axis.set_xlabel("Predicted")
            if column == 0:
                axis.set_ylabel(f"{endpoint}\nActual")
    FigureCanvasSVG(figure).print_svg(path)


def plot_history(histories, path, *, round_label="Training round"):
    figure = Figure(figsize=(11, 8), layout="constrained")
    axes = figure.subplots(3, 2).ravel()
    for axis, history in zip(axes, histories):
        rounds = np.arange(1, len(history["learn"]) + 1)
        axis.plot(rounds, history["learn"], label="Train")
        axis.plot(rounds, history["validation"], label="Validation")
        axis.axvline(history["best_round"], color="gray", linestyle="--", label="Best round")
        axis.set(title=f"Fold {history['fold']}", xlabel=round_label, ylabel="RMSE")
        axis.legend()
    for axis in axes[len(histories):]:
        axis.set_visible(False)
    FigureCanvasSVG(figure).print_svg(path)
