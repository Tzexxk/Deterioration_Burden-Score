"""Model-level SVG comparisons for assessment summaries."""

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure

from src.models.ehr_metrics import ENDPOINTS


def interval_plot(results, metric, path, *, title, xlabel):
    keys = list(results)
    models = sorted({key.split("/", 1)[0] for key in keys if "/" in key})
    palette = colormaps["tab20"].colors
    colors = dict(zip(models, (*palette[::2], *palette[1::2])))
    figure = Figure(figsize=(11, max(3.5, 1.8 + len(keys) * 0.31)), layout="constrained")
    figure.suptitle(title)
    axis, numbers = figure.subplots(1, 2, sharey=True, width_ratios=(3.3, 1.4))
    axis.set_yticks(range(len(keys)), keys)
    axis.set_ylim(len(keys) - 0.5, -0.5)
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", alpha=0.25)
    numbers.set_axis_off()
    numbers.set_title("Point estimate [95% CI]", loc="left", fontsize=10)
    relative = metric.endswith("_relative_pct")
    if relative:
        axis.axvline(0, color="gray", linestyle="--", linewidth=1)
    else:
        axis.set_xlim(-0.025, 1.025)
    for i, name in enumerate(keys):
        color = colors.get(name.split("/", 1)[0], "#606060")
        estimate = results[name][metric]
        point, lower, upper = (estimate[field] for field in ("point", "lower", "upper"))
        text = "N/A"
        if estimate["valid_draws"] and None not in (point, lower, upper):
            axis.hlines(i, lower, upper, color=color)
            axis.plot([lower, upper], [i, i], linestyle="none", marker="|", color=color)
            axis.scatter(point, i, color=color, s=18, zorder=3)
            text = (f"{point:+.2f}% [{lower:+.2f}%, {upper:+.2f}%]" if relative
                    else f"{point:.3f} [{lower:.3f}, {upper:.3f}]")
        numbers.text(0, i, text, va="center", fontsize=9, transform=numbers.get_yaxis_transform())
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
