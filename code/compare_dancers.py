from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


PERSON_0_CSV = Path("outputs/multi/features_0/window_feature_summary.csv")
PERSON_1_CSV = Path("outputs/multi/features_1/window_feature_summary.csv")

OUTPUT_DIR = Path("outputs/multi/comparison")


FEATURE_COLUMNS = [
    "movement_size_score",
    "dynamic_score",
    "smoothness_score",
    "stop_ratio",
]

# === Added constants ===
TARGET_PERSON_LABEL = "target"
LEARNER_PERSON_LABEL = "learner"
EPSILON = 1e-6

BODY_PARTS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

BODY_PART_METRICS = [
    "x_range",
    "y_range",
    "total_distance",
    "speed_std",
]

BODY_PART_FEATURE_COLUMNS = [
    f"{body_part}_{metric}"
    for body_part in BODY_PARTS
    for metric in BODY_PART_METRICS
]

ALL_COMPARISON_COLUMNS = FEATURE_COLUMNS + BODY_PART_FEATURE_COLUMNS


IMPROVEMENT_RULES = {
    "movement_size_score": "目標に比べて動きの大きさが不足している区間です。手首や足首を身体の中心から遠くへ出す意識を持つと、動きが大きく見えやすくなります。",
    "dynamic_score": "目標に比べて緩急が弱い区間です。速く動く部分と止める部分を分け、アクセントを明確にすることを意識してください。",
    "smoothness_score": "目標と比べて動きの鋭さや切り返し方が異なります。滑らかに見せたい場合は動作のつながりを、鋭く見せたい場合は止めのタイミングを意識してください。",
    "stop_ratio": "目標と比べて止めの時間が異なります。止まるべきポイントで身体を残す意識を持つと、振りの印象が近づきやすくなります。",
    "x_range": "この部位の横方向の軌道幅に差があります。目標の動きと比べて、左右方向にどこまで伸ばしているかを確認してください。",
    "y_range": "この部位の縦方向の軌道幅に差があります。目標の動きと比べて、上下方向の高さや沈み込みを確認してください。",
    "total_distance": "この部位の移動量に差があります。目標の軌道を確認し、同じ方向・同じ大きさで動かすことを意識してください。",
    "speed_std": "この部位の速度変化に差があります。目標のアクセント位置を確認し、加速と減速のタイミングを合わせてください。",
}


def load_data(
    person_0_csv: Path = PERSON_0_CSV,
    person_1_csv: Path = PERSON_1_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df0 = pd.read_csv(person_0_csv)
    df1 = pd.read_csv(person_1_csv)

    common_windows = sorted(
        set(df0["window_id"]).intersection(set(df1["window_id"]))
    )

    df0 = df0[df0["window_id"].isin(common_windows)].reset_index(drop=True)
    df1 = df1[df1["window_id"].isin(common_windows)].reset_index(drop=True)

    return df0, df1


# === Helper functions ===
def get_available_comparison_columns(df0: pd.DataFrame, df1: pd.DataFrame) -> list[str]:
    """Return feature columns available in both dataframes."""
    return [
        column
        for column in ALL_COMPARISON_COLUMNS
        if column in df0.columns and column in df1.columns
    ]


def calculate_relative_difference(target_value: float, learner_value: float) -> float:
    """Calculate learner's relative difference from target."""
    denominator = max(abs(target_value), EPSILON)
    return (learner_value - target_value) / denominator


def get_feature_group(feature: str) -> str:
    """Map feature name to a broad feedback rule group."""
    if feature in FEATURE_COLUMNS:
        return feature
    if feature.endswith("x_range"):
        return "x_range"
    if feature.endswith("y_range"):
        return "y_range"
    if feature.endswith("total_distance"):
        return "total_distance"
    if feature.endswith("speed_std"):
        return "speed_std"
    return feature


def get_body_part_name(feature: str) -> str:
    """Extract body part name from a feature name."""
    for metric in BODY_PART_METRICS:
        suffix = f"_{metric}"
        if feature.endswith(suffix):
            return feature[: -len(suffix)]
    return "overall"


def get_metric_label(feature: str) -> str:
    """Return a Japanese label for the metric represented by the feature."""
    feature_group = get_feature_group(feature)
    labels = {
        "x_range": "横方向の軌道幅",
        "y_range": "縦方向の軌道幅",
        "total_distance": "移動量",
        "speed_std": "緩急",
        "movement_size_score": "全体の動きの大きさ",
        "dynamic_score": "全体の緩急",
        "smoothness_score": "動きの鋭さ・滑らかさ",
        "stop_ratio": "停止時間の割合",
    }
    return labels.get(feature_group, feature_group)


def create_comparison_dataframe(df0: pd.DataFrame, df1: pd.DataFrame) -> pd.DataFrame:
    """Compare target person_0 and learner person_1 by time window."""
    rows = []
    comparison_columns = get_available_comparison_columns(df0, df1)

    for _, row0 in df0.iterrows():
        window_id = row0["window_id"]
        row1 = df1[df1["window_id"] == window_id].iloc[0]

        result = {
            "window_id": window_id,
            "start_sec": row0["start_sec"],
            "end_sec": row0["end_sec"],
        }

        max_feature = None
        max_relative_diff = -1.0

        for feature in comparison_columns:
            target_value = float(row0[feature])
            learner_value = float(row1[feature])
            absolute_diff = learner_value - target_value
            relative_diff = calculate_relative_difference(target_value, learner_value)

            result[f"{feature}_{TARGET_PERSON_LABEL}"] = target_value
            result[f"{feature}_{LEARNER_PERSON_LABEL}"] = learner_value
            result[f"{feature}_absolute_diff"] = absolute_diff
            result[f"{feature}_relative_diff"] = relative_diff

            if abs(relative_diff) > max_relative_diff:
                max_relative_diff = abs(relative_diff)
                max_feature = feature

        result["largest_difference_feature"] = max_feature
        result["largest_difference_relative_value"] = max_relative_diff
        result["largest_difference_percent"] = max_relative_diff * 100

        rows.append(result)

    return pd.DataFrame(rows)


def generate_comment(row: pd.Series) -> str:
    feature = row["largest_difference_feature"]
    target_value = row[f"{feature}_{TARGET_PERSON_LABEL}"]
    learner_value = row[f"{feature}_{LEARNER_PERSON_LABEL}"]
    relative_diff = row[f"{feature}_relative_diff"]
    percent = relative_diff * 100

    feature_group = get_feature_group(feature)
    body_part = get_body_part_name(feature)
    metric_label = get_metric_label(feature)

    suggestion = IMPROVEMENT_RULES.get(
        feature_group,
        "目標との差分が大きい特徴量です。該当区間の動画を見比べて、動きの方向・大きさ・タイミングを確認してください。",
    )

    if relative_diff > 0:
        direction_text = "学習者の方が目標より大きい値を示しています"
    else:
        direction_text = "学習者の方が目標より小さい値を示しています"

    if body_part == "overall":
        feature_text = metric_label
    else:
        feature_text = f"{body_part} の{metric_label}"

    return (
        f"{feature_text}に差があります。"
        f"目標={target_value:.3f}, 学習者={learner_value:.3f}, 差分={percent:+.1f}%です。"
        f"{direction_text}。"
        f"提案: {suggestion}"
    )


def add_comments(comparison_df: pd.DataFrame) -> pd.DataFrame:
    output_df = comparison_df.copy()
    output_df["comment"] = output_df.apply(generate_comment, axis=1)
    return output_df


def create_difference_plot(comparison_df: pd.DataFrame, output_path: Path) -> Path:
    """Plot relative differences of summary features between target and learner."""
    plt.figure(figsize=(12, 6))

    x = comparison_df["window_id"]

    for feature in FEATURE_COLUMNS:
        relative_column = f"{feature}_relative_diff"
        if relative_column not in comparison_df.columns:
            continue

        plt.plot(
            x,
            comparison_df[relative_column] * 100,
            marker="o",
            label=feature,
        )

    plt.axhline(0, linestyle="--")
    plt.xlabel("window_id")
    plt.ylabel("learner - target [% of target]")
    plt.title("Relative Feature Differences: Learner vs Target")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# === New function: create_body_part_difference_csv ===
def create_body_part_difference_csv(comparison_df: pd.DataFrame, output_path: Path) -> Path:
    """Export long-format body part differences for easier inspection."""
    rows = []

    for _, row in comparison_df.iterrows():
        for feature in BODY_PART_FEATURE_COLUMNS:
            relative_column = f"{feature}_relative_diff"
            target_column = f"{feature}_{TARGET_PERSON_LABEL}"
            learner_column = f"{feature}_{LEARNER_PERSON_LABEL}"

            if relative_column not in comparison_df.columns:
                continue

            rows.append(
                {
                    "window_id": row["window_id"],
                    "start_sec": row["start_sec"],
                    "end_sec": row["end_sec"],
                    "body_part": get_body_part_name(feature),
                    "metric": get_feature_group(feature),
                    "metric_label": get_metric_label(feature),
                    "feature": feature,
                    "target_value": row[target_column],
                    "learner_value": row[learner_column],
                    "absolute_diff": row[f"{feature}_absolute_diff"],
                    "relative_diff_percent": row[relative_column] * 100,
                }
            )

    body_part_df = pd.DataFrame(rows)
    body_part_df = body_part_df.sort_values(
        "relative_diff_percent",
        key=lambda series: series.abs(),
        ascending=False,
    )
    body_part_df.round(3).to_csv(output_path, index=False)
    return output_path


def create_report(comparison_df: pd.DataFrame, output_path: Path) -> Path:
    lines = []

    lines.append("Dance Comparison Report")
    lines.append("=" * 50)
    lines.append("person_0: target dancer")
    lines.append("person_1: learner dancer")
    lines.append("")

    largest_windows = comparison_df.sort_values(
        "largest_difference_relative_value",
        ascending=False,
    ).head(10)

    lines.append("[目標との差が大きい区間]")

    for _, row in largest_windows.iterrows():
        lines.append(
            f"{row['start_sec']:.1f}-{row['end_sec']:.1f}s | "
            f"feature={row['largest_difference_feature']} | "
            f"relative_diff={row['largest_difference_percent']:.1f}%"
        )

    lines.append("")
    lines.append("[比較指標の意味]")
    lines.append("- x_range: 各部位の横方向の軌道幅。左右方向にどれだけ大きく使えているかを示します。")
    lines.append("- y_range: 各部位の縦方向の軌道幅。上下方向の高さや沈み込みの大きさを示します。")
    lines.append("- total_distance: 各部位の総移動量。大きく動いたか、小刻みに多く動いたかを示します。")
    lines.append("- speed_std: 各部位の速度の標準偏差。値が大きいほど速度変化が大きく、緩急が強いことを示します。")

    lines.append("")
    lines.append("[区間ごとの改善提案]")

    for _, row in comparison_df.iterrows():
        lines.append(
            f"{row['start_sec']:.1f}-{row['end_sec']:.1f}s : {generate_comment(row)}"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_dancer_comparison(
    person_0_csv: Path = PERSON_0_CSV,
    person_1_csv: Path = PERSON_1_CSV,
    output_dir: Path = OUTPUT_DIR,
    decimals: int = 3,
    create_plot: bool = True,
) -> Dict[str, Path]:
    """Compare target dancer and learner dancer using window-level features.

    person_0 is treated as the target dancer.
    person_1 is treated as the learner dancer.
    Designed to be called from a Web app such as Streamlit.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df0, df1 = load_data(
        person_0_csv=person_0_csv,
        person_1_csv=person_1_csv,
    )

    comparison_df = create_comparison_dataframe(df0, df1)
    comparison_df = add_comments(comparison_df)

    comparison_csv = output_dir / "comparison_summary.csv"
    comparison_plot = output_dir / "comparison_plot.png"
    comparison_report = output_dir / "comparison_report.txt"
    body_part_difference_csv = output_dir / "body_part_difference_summary.csv"

    comparison_df.round(decimals).to_csv(comparison_csv, index=False)

    outputs: Dict[str, Path] = {
        "comparison_summary_csv": comparison_csv,
        "body_part_difference_csv": create_body_part_difference_csv(
            comparison_df,
            body_part_difference_csv,
        ),
        "comparison_report_txt": create_report(comparison_df, comparison_report),
    }

    if create_plot:
        outputs["comparison_plot"] = create_difference_plot(
            comparison_df,
            comparison_plot,
        )

    for output_path in outputs.values():
        print(f"Saved: {output_path}")

    return outputs



def main() -> None:
    run_dancer_comparison(
        person_0_csv=PERSON_0_CSV,
        person_1_csv=PERSON_1_CSV,
        output_dir=OUTPUT_DIR,
        decimals=3,
        create_plot=True,
    )


if __name__ == "__main__":
    main()