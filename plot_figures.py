from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
PROMPT_ONLY_PATH = BASE_DIR / "evaluation_outputs_to_score.csv"
RAG_FILES = {
    "k_3": BASE_DIR / "evaluation_ratings_K_3.csv",
    "k_5": BASE_DIR / "evaluation_ratings_K_5.csv",
    "k_7": BASE_DIR / "evaluation_ratings_K_7.csv",
}
RATING_DIMENSIONS = [
    "accuracy",
    "clarity",
    "safety",
    "groundedness",
    "citation_accuracy",
    "retrieval_relevance",
]
PROMPT_ONLY_METRICS = ["accuracy", "clarity", "safety"]
RAG_SPECIFIC_METRICS = ["groundedness", "citation_accuracy", "retrieval_relevance"]
CATEGORY_ORDER = ["typical", "varied", "edge_case", "rag_needed"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "clarity": "Clarity",
    "safety": "Safety",
    "groundedness": "Groundedness",
    "citation_accuracy": "Citation Accuracy",
    "retrieval_relevance": "Retrieval Relevance",
}
METRIC_COLORS = {
    "Accuracy": "#38bdf8",
    "Clarity": "#34d399",
    "Safety": "#f59e0b",
    "Groundedness": "#a78bfa",
    "Citation Accuracy": "#f472b6",
    "Retrieval Relevance": "#f87171",
}
APPROACH_COLORS = {
    "Prompting Only": "#f97316",
    "RAG Enhanced": "#38bdf8",
}
FIGURE_FACE = "#ffffff"
AXIS_FACE = "#ffffff"
TEXT_COLOR = "#111827"
GRID_COLOR = "#cbd5e1"
HEATMAP_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "heatmap_blue",
    ["#eff6ff", "#dbeafe", "#93c5fd", "#60a5fa"],
)


def apply_dark_style(ax):
    ax.set_facecolor(AXIS_FACE)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)


def finalize_figure(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(FIGURE_FACE)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def load_rag_ratings(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        next(file, None)
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            row = {"id": parts[0], "category": parts[1]}
            for metric, value in zip(RATING_DIMENSIONS, parts[-6:]):
                row[metric] = pd.to_numeric(value, errors="coerce")
            rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.dropna(subset=RATING_DIMENSIONS)


def load_prompt_only_model_b(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    output = frame[["id", "category"]].copy()
    for metric in PROMPT_ONLY_METRICS:
        output[metric] = pd.to_numeric(frame[f"{metric}_B"], errors="coerce")
    return output.dropna(subset=PROMPT_ONLY_METRICS)


def align_prompt_and_rag(prompt_frame: pd.DataFrame, rag_frame: pd.DataFrame):
    shared_ids = [case_id for case_id in prompt_frame["id"].tolist() if case_id in set(rag_frame["id"].tolist())]
    aligned_prompt = prompt_frame[prompt_frame["id"].isin(shared_ids)].copy()
    aligned_rag = rag_frame[rag_frame["id"].isin(shared_ids)].copy()
    aligned_prompt = aligned_prompt.set_index("id").loc[shared_ids].reset_index()
    aligned_rag = aligned_rag.set_index("id").loc[shared_ids].reset_index()
    return aligned_prompt, aligned_rag, shared_ids


def plot_ratings_per_query(rag_frame: pd.DataFrame, output_dir: Path):
    labels = rag_frame["id"].tolist()
    metrics = [METRIC_LABELS[m] for m in RATING_DIMENSIONS]
    x = list(range(len(labels)))
    width = 0.12

    fig, ax = plt.subplots(figsize=(18, 7))
    for idx, metric in enumerate(RATING_DIMENSIONS):
        offsets = [value + (idx - 2.5) * width for value in x]
        ax.bar(
            offsets,
            rag_frame[metric].tolist(),
            width=width,
            label=METRIC_LABELS[metric],
            color=METRIC_COLORS[METRIC_LABELS[metric]],
        )

    ax.set_title("Ratings Per Query")
    ax.set_xlabel("Query")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    legend = ax.legend(
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    apply_dark_style(ax)
    finalize_figure(fig, output_dir / "ratings_per_query.png")


def plot_rag_specific_metric_means(rag_frame: pd.DataFrame, output_dir: Path):
    metrics = [METRIC_LABELS[m] for m in RAG_SPECIFIC_METRICS]
    scores = [float(rag_frame[m].mean()) for m in RAG_SPECIFIC_METRICS]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(metrics, scores, color=[METRIC_COLORS[m] for m in metrics])
    ax.set_title("RAG-Specific Metric Means")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mean Score")
    ax.set_ylim(0, 5)
    apply_dark_style(ax)
    finalize_figure(fig, output_dir / "rag_specific_metric_means.png")


def plot_rag_specific_score_distribution(rag_frame: pd.DataFrame, output_dir: Path):
    data = [rag_frame[m].tolist() for m in RAG_SPECIFIC_METRICS]
    labels = [METRIC_LABELS[m] for m in RAG_SPECIFIC_METRICS]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    box = ax.boxplot(data, patch_artist=True, labels=labels)
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(METRIC_COLORS[label])
        patch.set_alpha(0.8)
    for median in box["medians"]:
        median.set_color("#ffffff")
    ax.set_title("RAG-Specific Score Distribution Across Test Cases")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 5)
    apply_dark_style(ax)
    finalize_figure(fig, output_dir / "rag_specific_score_distribution_across_test_cases.png")


def plot_retrieval_quality_vs_answer_accuracy(rag_frame: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    intensity = (rag_frame["retrieval_relevance"] + rag_frame["accuracy"]) / 2.0
    scatter = ax.scatter(
        rag_frame["retrieval_relevance"],
        rag_frame["accuracy"],
        c=intensity,
        cmap="viridis",
        vmin=1,
        vmax=5,
        s=80,
        alpha=0.9,
        edgecolors="#111827",
        linewidths=0.6,
    )

    ax.set_title("Retrieval Quality vs Answer Accuracy")
    ax.set_xlabel("Retrieval Relevance Score")
    ax.set_ylabel("Answer Accuracy Score")
    ax.set_xlim(0.8, 5.1)
    ax.set_ylim(0.8, 5.1)
    ax.plot([1, 5], [1, 5], linestyle="--", color="#94a3b8", linewidth=1)
    apply_dark_style(ax)
    cbar = fig.colorbar(scatter, ax=ax, label="Mean score intensity (1-5)")
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT_COLOR)
    cbar.ax.yaxis.label.set_color(TEXT_COLOR)
    cbar.outline.set_edgecolor(TEXT_COLOR)
    finalize_figure(fig, output_dir / "retrieval_quality_vs_answer_accuracy.png")


def plot_rag_specific_mean_by_category_heatmap(rag_frame: pd.DataFrame, output_dir: Path):
    available_categories = [cat for cat in CATEGORY_ORDER if cat in set(rag_frame["category"].tolist())]
    if not available_categories:
        return
    matrix = []
    for category in available_categories:
        category_rows = rag_frame[rag_frame["category"] == category]
        matrix.append([float(category_rows[m].mean()) for m in RAG_SPECIFIC_METRICS])

    fig, ax = plt.subplots(figsize=(8, max(4, len(available_categories) * 1.2)))
    im = ax.imshow(matrix, cmap=HEATMAP_BLUE_CMAP, vmin=1, vmax=5, aspect="auto")
    ax.set_title("RAG-Specific Mean Score By Test Case Category")
    ax.set_xticks(range(len(RAG_SPECIFIC_METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in RAG_SPECIFIC_METRICS], rotation=20, ha="right")
    ax.set_yticks(range(len(available_categories)))
    ax.set_yticklabels(available_categories)
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=TEXT_COLOR)
    apply_dark_style(ax)
    cbar = fig.colorbar(im, ax=ax, label="Mean score (1-5)")
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT_COLOR)
    cbar.outline.set_edgecolor(TEXT_COLOR)
    finalize_figure(fig, output_dir / "rag_specific_mean_score_by_category.png")


def plot_prompting_vs_rag(prompt_frame: pd.DataFrame, rag_frame: pd.DataFrame, output_dir: Path):
    labels = [METRIC_LABELS[m] for m in PROMPT_ONLY_METRICS]
    prompt_scores = [float(prompt_frame[m].mean()) for m in PROMPT_ONLY_METRICS]
    rag_scores = [float(rag_frame[m].mean()) for m in PROMPT_ONLY_METRICS]
    x = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        [value - width / 2 for value in x],
        prompt_scores,
        width=width,
        label="Prompting Only",
        color=APPROACH_COLORS["Prompting Only"],
    )
    ax.bar(
        [value + width / 2 for value in x],
        rag_scores,
        width=width,
        label="RAG Enhanced",
        color=APPROACH_COLORS["RAG Enhanced"],
    )
    ax.set_title("Prompting Only vs RAG Enhanced")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mean Score")
    ax.set_ylim(0, 5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    legend = ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    apply_dark_style(ax)
    finalize_figure(fig, output_dir / "prompting_only_vs_rag_enhanced.png")


def plot_overall_distribution(prompt_frame: pd.DataFrame, rag_frame: pd.DataFrame, output_dir: Path):
    prompt_overall = prompt_frame[PROMPT_ONLY_METRICS].mean(axis=1).tolist()
    rag_overall = rag_frame[PROMPT_ONLY_METRICS].mean(axis=1).tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    box = ax.boxplot(
        [prompt_overall, rag_overall],
        patch_artist=True,
        labels=["Prompting Only", "RAG Enhanced"],
    )
    for patch, label in zip(box["boxes"], ["Prompting Only", "RAG Enhanced"]):
        patch.set_facecolor(APPROACH_COLORS[label])
        patch.set_alpha(0.85)
    for median in box["medians"]:
        median.set_color("#ffffff")
    ax.set_title("Overall Score Distribution Across Shared Test Cases")
    ax.set_ylabel("Overall Score")
    ax.set_ylim(0, 5)
    apply_dark_style(ax)
    finalize_figure(fig, output_dir / "overall_score_distribution_across_test_cases.png")


def plot_mean_overall_by_category_heatmap(prompt_frame: pd.DataFrame, rag_frame: pd.DataFrame, output_dir: Path):
    rows = []
    row_labels = []
    for category in CATEGORY_ORDER:
        prompt_rows = prompt_frame[prompt_frame["category"] == category]
        rag_rows = rag_frame[rag_frame["category"] == category]
        if prompt_rows.empty and rag_rows.empty:
            continue
        row_labels.append(category)
        rows.append(
            [
                float(prompt_rows[PROMPT_ONLY_METRICS].mean(axis=1).mean()) if not prompt_rows.empty else float("nan"),
                float(rag_rows[PROMPT_ONLY_METRICS].mean(axis=1).mean()) if not rag_rows.empty else float("nan"),
            ]
        )

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(7, max(4, len(row_labels) * 1.2)))
    im = ax.imshow(rows, cmap=HEATMAP_BLUE_CMAP, vmin=1, vmax=5, aspect="auto")
    ax.set_title("Mean Overall Score By Test Case Category")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Prompting Only", "RAG Enhanced"])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for row_idx, row in enumerate(rows):
        for col_idx, value in enumerate(row):
            label = "N/A" if pd.isna(value) else f"{value:.2f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color=TEXT_COLOR)
    apply_dark_style(ax)
    cbar = fig.colorbar(im, ax=ax, label="Mean overall score (1-5)")
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT_COLOR)
    cbar.outline.set_edgecolor(TEXT_COLOR)
    finalize_figure(fig, output_dir / "mean_overall_score_by_category.png")


def build_all_figures_for_k(k_label: str, rag_path: Path, prompt_frame: pd.DataFrame):
    output_dir = FIGURES_DIR / k_label
    rag_frame = load_rag_ratings(rag_path)

    plot_ratings_per_query(rag_frame, output_dir)
    plot_rag_specific_metric_means(rag_frame, output_dir)
    plot_rag_specific_score_distribution(rag_frame, output_dir)
    plot_retrieval_quality_vs_answer_accuracy(rag_frame, output_dir)
    plot_rag_specific_mean_by_category_heatmap(rag_frame, output_dir)

    aligned_prompt, aligned_rag, shared_ids = align_prompt_and_rag(prompt_frame, rag_frame)
    if not shared_ids:
        return
    plot_prompting_vs_rag(aligned_prompt, aligned_rag, output_dir)
    plot_overall_distribution(aligned_prompt, aligned_rag, output_dir)
    plot_mean_overall_by_category_heatmap(aligned_prompt, aligned_rag, output_dir)


def main():
    prompt_frame = load_prompt_only_model_b(PROMPT_ONLY_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for k_label, rag_path in RAG_FILES.items():
        build_all_figures_for_k(k_label, rag_path, prompt_frame)
    print(f"Saved matplotlib figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
