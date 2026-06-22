
"""
Create a difference timeline for dance motion comparison.

Inputs:
    outputs/multi/angle_comparison/angle_difference_summary.csv
    outputs/multi/comparison/body_part_difference_summary.csv

Outputs:
    outputs/multi/timeline/difference_timeline.csv
    outputs/multi/timeline/top_feedback_points.csv
    outputs/multi/timeline/difference_timeline.png
    outputs/multi/timeline/difference_timeline_report.txt

Run:
    python code/difference_timeline.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


ANGLE_SUMMARY_CSV = Path("outputs/multi/angle_comparison/angle_difference_summary.csv")
BODY_PART_DIFF_CSV = Path("outputs/multi/comparison/body_part_difference_summary.csv")
OUTPUT_DIR = Path("outputs/multi/timeline")

TOP_N = 10
ANGLE_IMPORTANCE_NORMALIZER = 45.0
BODY_PART_IMPORTANCE_NORMALIZER = 100.0

ANGLE_LABELS = {
    "left_elbow_angle": "左肘",
    "right_elbow_angle": "右肘",
    "left_knee_angle": "左膝",
    "right_knee_angle": "右膝",
    "left_shoulder_angle": "左肩",
    "right_shoulder_angle": "右肩",
    "left_hip_angle": "左股関節",
    "right_hip_angle": "右股関節",
    "torso_lean_angle": "胴体の傾き",
}

BODY_PART_LABELS = {
    "left_shoulder": "左肩",
    "right_shoulder": "右肩",
    "left_elbow": "左肘",
    "right_elbow": "右肘",
    "left_wrist": "左手首",
    "right_wrist": "右手首",
    "left_hip": "左腰",
    "right_hip": "右腰",
    "left_knee": "左膝",
    "right_knee": "右膝",
    "left_ankle": "左足首",
    "right_ankle": "右足首",
}

METRIC_LABELS = {
    "x_range": "横方向の軌道幅",
    "y_range": "縦方向の軌道幅",
    "total_distance": "移動量",
    "speed_std": "緩急",
}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def get_time_label(start_sec: float, end_sec: float) -> str:
    return f"{start_sec:.1f}-{end_sec:.1f}s"


def normalize_score(value: float, normalizer: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(min(abs(value) / normalizer, 1.0) * 100.0)


def create_angle_comment(body_part: str, diff_deg: float) -> str:
    return (
        f"{body_part}の角度差が大きい区間です。"
        f"目標との差は約{diff_deg:.1f}°です。"
        "該当時間でポーズを止めて、関節の曲げ伸ばしや上半身の向きを見比べてください。"
    )


def create_body_part_comment(
    body_part_label: str,
    metric_label: str,
    metric_key: str,
    relative_diff_percent: float,
) -> str:
    if relative_diff_percent > 0:
        direction_text = "学習者の方が目標より大きい"
    else:
        direction_text = "学習者の方が目標より小さい"

    if metric_key in ["x_range", "y_range"]:
        action_text = "軌道の幅や到達位置を目標と見比べてください。"
    elif metric_key == "total_distance":
        action_text = "同じ方向にどれだけ大きく動かしているかを確認してください。"
    elif metric_key == "speed_std":
        action_text = "加速・減速・止めのタイミングを目標と合わせてください。"
    else:
        action_text = "該当時間の動きを見比べてください。"

    return (
        f"{body_part_label}の{metric_label}に差があります。"
        f"{direction_text}値を示しており、差分は{relative_diff_percent:+.1f}%です。"
        f"{action_text}"
    )


def build_angle_events(angle_df: pd.DataFrame) -> pd.DataFrame:
    """Create timeline events from angle difference summary."""
    rows = []

    for _, row in angle_df.iterrows():
        angle_name = row.get("largest_angle_difference")
        if pd.isna(angle_name):
            continue

        diff_deg = float(row.get("largest_angle_mean_abs_diff_deg", 0.0))
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        body_part = ANGLE_LABELS.get(str(angle_name), str(angle_name))
        importance_score = normalize_score(diff_deg, ANGLE_IMPORTANCE_NORMALIZER)

        rows.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "time_range": get_time_label(start_sec, end_sec),
                "event_type": "angle",
                "body_part": body_part,
                "metric": "角度",
                "raw_value": diff_deg,
                "raw_unit": "deg",
                "importance_score": importance_score,
                "comment": create_angle_comment(body_part, diff_deg),
            }
        )

    return pd.DataFrame(rows)


def build_body_part_events(body_df: pd.DataFrame) -> pd.DataFrame:
    """Create trajectory and dynamics events from body part difference summary."""
    rows = []

    for _, row in body_df.iterrows():
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        body_part_key = str(row.get("body_part", ""))
        metric_key = str(row.get("metric", ""))
        relative_diff_percent = float(row.get("relative_diff_percent", 0.0))

        if metric_key in ["x_range", "y_range", "total_distance"]:
            event_type = "trajectory"
        elif metric_key == "speed_std":
            event_type = "dynamics"
        else:
            continue

        body_part_label = BODY_PART_LABELS.get(body_part_key, body_part_key)
        metric_label = METRIC_LABELS.get(metric_key, metric_key)
        importance_score = normalize_score(relative_diff_percent, BODY_PART_IMPORTANCE_NORMALIZER)

        rows.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "time_range": get_time_label(start_sec, end_sec),
                "event_type": event_type,
                "body_part": body_part_label,
                "metric": metric_label,
                "raw_value": relative_diff_percent,
                "raw_unit": "%",
                "importance_score": importance_score,
                "comment": create_body_part_comment(
                    body_part_label=body_part_label,
                    metric_label=metric_label,
                    metric_key=metric_key,
                    relative_diff_percent=relative_diff_percent,
                ),
            }
        )

    return pd.DataFrame(rows)


def create_difference_timeline(
    angle_summary_csv: Path = ANGLE_SUMMARY_CSV,
    body_part_difference_csv: Path = BODY_PART_DIFF_CSV,
) -> pd.DataFrame:
    angle_df = load_csv(angle_summary_csv)
    body_df = load_csv(body_part_difference_csv)

    angle_events = build_angle_events(angle_df)
    body_events = build_body_part_events(body_df)

    timeline_df = pd.concat([angle_events, body_events], ignore_index=True)
    if timeline_df.empty:
        return timeline_df

    timeline_df = timeline_df.sort_values(
        ["importance_score", "start_sec"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return timeline_df


def create_top_feedback_points(timeline_df: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Select top feedback points while reducing duplicates from the same time and body part."""
    if timeline_df.empty:
        return timeline_df

    selected_rows = []
    used_keys = set()

    for _, row in timeline_df.iterrows():
        key = (row["time_range"], row["body_part"], row["event_type"])
        if key in used_keys:
            continue

        selected_rows.append(row)
        used_keys.add(key)

        if len(selected_rows) >= top_n:
            break

    return pd.DataFrame(selected_rows)


def plot_difference_timeline(timeline_df: pd.DataFrame, output_path: Path) -> Path:
    """Plot top timeline events as a simple readable scatter timeline."""
    if timeline_df.empty:
        return output_path

    plot_df = timeline_df.sort_values("start_sec").copy()
    event_type_to_y = {
        "angle": 3,
        "trajectory": 2,
        "dynamics": 1,
    }
    plot_df["y"] = plot_df["event_type"].map(event_type_to_y)

    plt.figure(figsize=(13, 5))

    for event_type, group_df in plot_df.groupby("event_type"):
        plt.scatter(
            group_df["start_sec"],
            group_df["y"],
            s=group_df["importance_score"].clip(lower=10) * 4,
            alpha=0.7,
            label=event_type,
        )

    top_labels = plot_df.sort_values("importance_score", ascending=False).head(8)
    for _, row in top_labels.iterrows():
        label = f"{row['time_range']}\n{row['body_part']} {row['metric']}"
        plt.annotate(
            label,
            (row["start_sec"], row["y"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    plt.yticks([1, 2, 3], ["Dynamics", "Trajectory", "Angle"])
    plt.xlabel("time [s]")
    plt.ylabel("difference type")
    plt.title("Difference Timeline")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def create_text_report(top_feedback_df: pd.DataFrame, output_path: Path) -> Path:
    lines = []
    lines.append("Difference Timeline Report")
    lines.append("=" * 50)
    lines.append("")
    lines.append("[今回優先して直すポイント]")

    if top_feedback_df.empty:
        lines.append("- 大きな差分イベントは検出されませんでした。")
    else:
        for index, (_, row) in enumerate(top_feedback_df.iterrows(), start=1):
            lines.append(
                f"{index}. {row['time_range']} | {row['body_part']} | {row['metric']} | "
                f"score={row['importance_score']:.1f}"
            )
            lines.append(f"   {row['comment']}")

    lines.append("")
    lines.append("[指標の見方]")
    lines.append("- angle: その瞬間のポーズ・関節角度の違いを示します。")
    lines.append("- trajectory: 手足などの軌道幅・移動量の違いを示します。")
    lines.append("- dynamics: 速度変化、つまり緩急やアクセントの違いを示します。")
    lines.append("- importance_score: 差分の大きさを0〜100程度に正規化した目安です。")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_difference_timeline(
    angle_summary_csv: Path = ANGLE_SUMMARY_CSV,
    body_part_difference_csv: Path = BODY_PART_DIFF_CSV,
    output_dir: Path = OUTPUT_DIR,
    top_n: int = TOP_N,
    decimals: int = 3,
    create_plot: bool = True,
) -> Dict[str, Path]:
    """Create timeline events and top feedback points for the Web app."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timeline_csv = output_dir / "difference_timeline.csv"
    top_feedback_csv = output_dir / "top_feedback_points.csv"
    timeline_plot = output_dir / "difference_timeline.png"
    report_txt = output_dir / "difference_timeline_report.txt"

    timeline_df = create_difference_timeline(
        angle_summary_csv=angle_summary_csv,
        body_part_difference_csv=body_part_difference_csv,
    )
    top_feedback_df = create_top_feedback_points(timeline_df, top_n=top_n)

    timeline_df.round(decimals).to_csv(timeline_csv, index=False)
    top_feedback_df.round(decimals).to_csv(top_feedback_csv, index=False)

    outputs: Dict[str, Path] = {
        "difference_timeline_csv": timeline_csv,
        "top_feedback_points_csv": top_feedback_csv,
        "difference_timeline_report_txt": create_text_report(top_feedback_df, report_txt),
    }

    if create_plot:
        outputs["difference_timeline_plot"] = plot_difference_timeline(
            top_feedback_df,
            timeline_plot,
        )

    for output_path in outputs.values():
        print(f"Saved: {output_path}")

    return outputs


def main() -> None:
    run_difference_timeline(
        angle_summary_csv=ANGLE_SUMMARY_CSV,
        body_part_difference_csv=BODY_PART_DIFF_CSV,
        output_dir=OUTPUT_DIR,
        top_n=TOP_N,
        decimals=3,
        create_plot=True,
    )


if __name__ == "__main__":
    main()
