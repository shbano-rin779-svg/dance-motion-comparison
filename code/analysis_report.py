

"""
Generate visualizations and simple text feedback from window_feature_summary.csv.

Inputs:
    outputs/features/window_feature_summary.csv

Outputs:
    outputs/report/window_score_timeseries.png
    outputs/report/window_feature_comments.csv
    outputs/report/analysis_report.txt

Run:
    python code/analysis_report.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV_PATH = Path("outputs/features/window_feature_summary.csv")
OUTPUT_DIR = Path("outputs/report")

SCORE_COLUMNS = [
    "movement_size_score",
    "dynamic_score",
    "smoothness_score",
    "stop_ratio",
]


def load_window_features(csv_path: Path) -> pd.DataFrame:
    """Load window-level feature summary."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = ["window_id", "start_sec", "end_sec", *SCORE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def add_time_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable time range label."""
    labeled_df = df.copy()
    labeled_df["time_range"] = labeled_df.apply(
        lambda row: f"{row['start_sec']:.1f}-{row['end_sec']:.1f}s",
        axis=1,
    )
    return labeled_df


def get_level(value: float, low_threshold: float, high_threshold: float) -> str:
    """Classify a feature value into low / medium / high."""
    if pd.isna(value):
        return "unknown"
    if value >= high_threshold:
        return "high"
    if value <= low_threshold:
        return "low"
    return "medium"


def add_feature_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Add low / medium / high labels based on quantiles."""
    labeled_df = df.copy()

    for column in SCORE_COLUMNS:
        low_threshold = labeled_df[column].quantile(0.33)
        high_threshold = labeled_df[column].quantile(0.67)
        labeled_df[f"{column}_level"] = labeled_df[column].apply(
            lambda value: get_level(value, low_threshold, high_threshold)
        )

    return labeled_df


def generate_comment(row: pd.Series) -> str:
    """Generate a simple feedback comment for one time window."""
    comments = []

    movement_level = row["movement_size_score_level"]
    dynamic_level = row["dynamic_score_level"]
    smoothness_level = row["smoothness_score_level"]
    stop_level = row["stop_ratio_level"]

    if movement_level == "high":
        comments.append("手足を大きく使えている区間です")
    elif movement_level == "low":
        comments.append("動きの大きさは控えめな区間です")

    if dynamic_level == "high":
        comments.append("速度変化が大きく、緩急が強く出ています")
    elif dynamic_level == "low":
        comments.append("速度変化が小さく、動きが一定になりやすい区間です")

    if smoothness_level == "high":
        comments.append("jerkが大きく、急な切り返しや鋭い動きが多い可能性があります")
    elif smoothness_level == "low":
        comments.append("jerkが小さく、比較的なめらかな動きの区間です")

    if stop_level == "high":
        comments.append("停止している時間が長く、止めの動きが含まれている可能性があります")
    elif stop_level == "low":
        comments.append("停止時間が短く、動き続けている区間です")

    if not comments:
        return "全体として中程度の特徴を示す区間です。"

    return "。".join(comments) + "。"


def add_comments(df: pd.DataFrame) -> pd.DataFrame:
    """Add generated comments to the dataframe."""
    commented_df = df.copy()
    commented_df["comment"] = commented_df.apply(generate_comment, axis=1)
    return commented_df


def plot_window_scores(df: pd.DataFrame, output_path: Path) -> None:
    """Plot time-series of key window-level scores."""
    plt.figure(figsize=(12, 6))

    x = (df["start_sec"] + df["end_sec"]) / 2

    for column in ["movement_size_score", "dynamic_score", "smoothness_score"]:
        values = df[column].astype(float)
        if values.max() == values.min():
            normalized_values = values * 0
        else:
            normalized_values = (values - values.min()) / (values.max() - values.min())
        plt.plot(x, normalized_values, marker="o", linewidth=1.5, label=f"{column} (normalized)")

    plt.xlabel("time [s]")
    plt.ylabel("normalized score")
    plt.title("Window-level dance feature scores")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def create_text_report(df: pd.DataFrame, output_path: Path) -> None:
    """Create a readable text report from window-level comments."""
    lines = []
    lines.append("Dance analysis report")
    lines.append("=" * 50)
    lines.append("")

    top_movement = df.sort_values("movement_size_score", ascending=False).head(3)
    top_dynamic = df.sort_values("dynamic_score", ascending=False).head(3)
    top_smoothness = df.sort_values("smoothness_score", ascending=True).head(3)

    lines.append("[動きが大きい区間]")
    for _, row in top_movement.iterrows():
        lines.append(
            f"- {row['time_range']}: movement_size_score={row['movement_size_score']:.3f}"
        )
    lines.append("")

    lines.append("[緩急が強い区間]")
    for _, row in top_dynamic.iterrows():
        lines.append(
            f"- {row['time_range']}: dynamic_score={row['dynamic_score']:.3f}"
        )
    lines.append("")

    lines.append("[比較的なめらかな区間]")
    for _, row in top_smoothness.iterrows():
        lines.append(
            f"- {row['time_range']}: smoothness_score={row['smoothness_score']:.3f}"
        )
    lines.append("")

    lines.append("[区間ごとの簡易コメント]")
    for _, row in df.iterrows():
        lines.append(f"- {row['time_range']}: {row['comment']}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def reorder_comment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Place readable columns at the beginning of the output CSV."""
    first_columns = [
        "window_id",
        "start_sec",
        "end_sec",
        "time_range",
        "movement_size_score",
        "dynamic_score",
        "smoothness_score",
        "stop_ratio",
        "movement_size_score_level",
        "dynamic_score_level",
        "smoothness_score_level",
        "stop_ratio_level",
        "comment",
    ]
    existing_first_columns = [column for column in first_columns if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in existing_first_columns]
    return df[existing_first_columns + remaining_columns]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    window_features = load_window_features(INPUT_CSV_PATH)
    window_features = add_time_label(window_features)
    window_features = add_feature_levels(window_features)
    window_features = add_comments(window_features)

    plot_path = OUTPUT_DIR / "window_score_timeseries.png"
    comments_csv_path = OUTPUT_DIR / "window_feature_comments.csv"
    report_txt_path = OUTPUT_DIR / "analysis_report.txt"

    plot_window_scores(window_features, plot_path)

    output_comments = reorder_comment_columns(window_features)
    output_comments.to_csv(comments_csv_path, index=False)

    create_text_report(window_features, report_txt_path)

    print(f"Saved: {plot_path}")
    print(f"Saved: {comments_csv_path}")
    print(f"Saved: {report_txt_path}")


if __name__ == "__main__":
    main()