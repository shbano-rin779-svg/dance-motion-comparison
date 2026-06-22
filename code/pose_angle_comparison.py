"""
Compare pose angles between target dancer and learner dancer.

Inputs:
    outputs/multi/person_0_pose.csv  # target dancer
    outputs/multi/person_1_pose.csv  # learner dancer

Outputs:
    outputs/multi/angle_comparison/angle_difference_summary.csv
    outputs/multi/angle_comparison/angle_difference_events.csv
    outputs/multi/angle_comparison/angle_difference_timeseries.png
    outputs/multi/angle_comparison/angle_difference_report.txt

Run:
    python code/pose_angle_comparison.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_CSV_PATH = Path("outputs/multi/person_0_pose.csv")
LEARNER_CSV_PATH = Path("outputs/multi/person_1_pose.csv")
OUTPUT_DIR = Path("outputs/multi/angle_comparison")

WINDOW_SEC = 1.0
ANGLE_DIFF_THRESHOLD_DEG = 25.0
MIN_VALID_FRAMES_PER_WINDOW = 5

# angle_name: (point_a, vertex_point, point_c)
# angle is calculated at vertex_point.
ANGLE_DEFINITIONS: Dict[str, Tuple[str, str, str]] = {
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "left_shoulder_angle": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder_angle": ("right_hip", "right_shoulder", "right_elbow"),
    "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
}

ANGLE_LABELS = {
    "left_elbow_angle": "左肘の角度",
    "right_elbow_angle": "右肘の角度",
    "left_knee_angle": "左膝の角度",
    "right_knee_angle": "右膝の角度",
    "left_shoulder_angle": "左肩の角度",
    "right_shoulder_angle": "右肩の角度",
    "left_hip_angle": "左股関節の角度",
    "right_hip_angle": "右股関節の角度",
    "torso_lean_angle": "胴体の傾き",
}

FEEDBACK_RULES = {
    "elbow": "腕の曲げ伸ばしに差があります。目標の肘の開き具合を確認し、腕を伸ばす／曲げるタイミングを合わせてください。",
    "knee": "膝の曲げ伸ばしに差があります。沈み込みやステップ時の膝の使い方を目標と見比べてください。",
    "shoulder": "肩まわりの形に差があります。肘の位置だけでなく、肩から腕を出す方向を確認してください。",
    "hip": "股関節まわりの形に差があります。上半身と脚の角度、体重の乗せ方を確認してください。",
    "torso": "胴体の傾きに差があります。上半身の倒し方や重心の位置を目標と見比べてください。",
}


def load_pose_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for column in df.columns:
        if column != "pose_detected":
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["pose_detected"] = pd.to_numeric(df["pose_detected"], errors="coerce").fillna(0).astype(int)
    return df


def get_joint_xy(row: pd.Series, joint_name: str) -> np.ndarray | None:
    x = row.get(f"{joint_name}_x")
    y = row.get(f"{joint_name}_y")
    visibility = row.get(f"{joint_name}_visibility", 1.0)

    if pd.isna(x) or pd.isna(y) or pd.isna(visibility):
        return None
    if visibility < 0.5:
        return None

    return np.array([float(x), float(y)], dtype=float)


def calculate_three_point_angle(point_a: np.ndarray, vertex: np.ndarray, point_c: np.ndarray) -> float:
    vector_a = point_a - vertex
    vector_c = point_c - vertex

    norm_a = np.linalg.norm(vector_a)
    norm_c = np.linalg.norm(vector_c)
    if norm_a == 0 or norm_c == 0:
        return np.nan

    cos_angle = np.dot(vector_a, vector_c) / (norm_a * norm_c)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def calculate_torso_lean_angle(row: pd.Series) -> float:
    left_shoulder = get_joint_xy(row, "left_shoulder")
    right_shoulder = get_joint_xy(row, "right_shoulder")
    left_hip = get_joint_xy(row, "left_hip")
    right_hip = get_joint_xy(row, "right_hip")

    if any(point is None for point in [left_shoulder, right_shoulder, left_hip, right_hip]):
        return np.nan

    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    torso_vector = shoulder_center - hip_center

    if np.linalg.norm(torso_vector) == 0:
        return np.nan

    # Angle from vertical direction in image coordinates.
    vertical_vector = np.array([0.0, -1.0])
    cos_angle = np.dot(torso_vector, vertical_vector) / (
        np.linalg.norm(torso_vector) * np.linalg.norm(vertical_vector)
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cos_angle)))

    # Add sign using horizontal component.
    if torso_vector[0] < 0:
        angle *= -1

    return angle


def calculate_angles_for_row(row: pd.Series) -> Dict[str, float]:
    angles: Dict[str, float] = {}

    if row["pose_detected"] != 1:
        for angle_name in ANGLE_DEFINITIONS:
            angles[angle_name] = np.nan
        angles["torso_lean_angle"] = np.nan
        return angles

    for angle_name, (joint_a, joint_b, joint_c) in ANGLE_DEFINITIONS.items():
        point_a = get_joint_xy(row, joint_a)
        point_b = get_joint_xy(row, joint_b)
        point_c = get_joint_xy(row, joint_c)

        if point_a is None or point_b is None or point_c is None:
            angles[angle_name] = np.nan
            continue

        angles[angle_name] = calculate_three_point_angle(point_a, point_b, point_c)

    angles["torso_lean_angle"] = calculate_torso_lean_angle(row)
    return angles


def add_angle_columns(df: pd.DataFrame) -> pd.DataFrame:
    angle_rows = []
    for _, row in df.iterrows():
        angle_rows.append(calculate_angles_for_row(row))

    angle_df = pd.DataFrame(angle_rows)
    return pd.concat([df.reset_index(drop=True), angle_df], axis=1)


def merge_target_and_learner(target_df: pd.DataFrame, learner_df: pd.DataFrame) -> pd.DataFrame:
    angle_columns = list(ANGLE_DEFINITIONS.keys()) + ["torso_lean_angle"]
    base_columns = ["frame_index", "timestamp_sec", "pose_detected", *angle_columns]

    target_part = target_df[base_columns].rename(
        columns={column: f"{column}_target" for column in base_columns if column not in ["frame_index", "timestamp_sec"]}
    )
    learner_part = learner_df[base_columns].rename(
        columns={column: f"{column}_learner" for column in base_columns if column not in ["frame_index", "timestamp_sec"]}
    )

    merged_df = pd.merge(
        target_part,
        learner_part,
        on=["frame_index", "timestamp_sec"],
        how="inner",
    )

    for angle_name in angle_columns:
        merged_df[f"{angle_name}_diff_abs"] = (
            merged_df[f"{angle_name}_learner"] - merged_df[f"{angle_name}_target"]
        ).abs()
        merged_df[f"{angle_name}_diff_signed"] = (
            merged_df[f"{angle_name}_learner"] - merged_df[f"{angle_name}_target"]
        )

    return merged_df


def summarize_angle_differences_by_window(
    merged_df: pd.DataFrame,
    window_sec: float = WINDOW_SEC,
    min_valid_frames_per_window: int = MIN_VALID_FRAMES_PER_WINDOW,
) -> pd.DataFrame:
    angle_columns = list(ANGLE_DEFINITIONS.keys()) + ["torso_lean_angle"]
    max_time = float(merged_df["timestamp_sec"].max())
    rows = []

    window_id = 0
    start_time = 0.0
    while start_time < max_time:
        end_time = start_time + window_sec
        window_df = merged_df[
            (merged_df["timestamp_sec"] >= start_time)
            & (merged_df["timestamp_sec"] < end_time)
        ].copy()

        valid_pose = (
            (window_df["pose_detected_target"] == 1)
            & (window_df["pose_detected_learner"] == 1)
        )
        window_df = window_df[valid_pose]

        if len(window_df) < min_valid_frames_per_window:
            window_id += 1
            start_time = end_time
            continue

        result = {
            "window_id": window_id,
            "start_sec": start_time,
            "end_sec": end_time,
            "valid_frames": len(window_df),
        }

        max_angle_name = None
        max_angle_diff = -1.0

        for angle_name in angle_columns:
            abs_diff_col = f"{angle_name}_diff_abs"
            signed_diff_col = f"{angle_name}_diff_signed"

            mean_abs_diff = float(window_df[abs_diff_col].mean(skipna=True))
            max_abs_diff = float(window_df[abs_diff_col].max(skipna=True))
            mean_signed_diff = float(window_df[signed_diff_col].mean(skipna=True))

            result[f"{angle_name}_mean_abs_diff_deg"] = mean_abs_diff
            result[f"{angle_name}_max_abs_diff_deg"] = max_abs_diff
            result[f"{angle_name}_mean_signed_diff_deg"] = mean_signed_diff

            if not np.isnan(mean_abs_diff) and mean_abs_diff > max_angle_diff:
                max_angle_diff = mean_abs_diff
                max_angle_name = angle_name

        result["largest_angle_difference"] = max_angle_name
        result["largest_angle_mean_abs_diff_deg"] = max_angle_diff
        rows.append(result)

        window_id += 1
        start_time = end_time

    return pd.DataFrame(rows)


def get_angle_group(angle_name: str) -> str:
    if "elbow" in angle_name:
        return "elbow"
    if "knee" in angle_name:
        return "knee"
    if "shoulder" in angle_name:
        return "shoulder"
    if "hip" in angle_name:
        return "hip"
    if "torso" in angle_name:
        return "torso"
    return "default"


def generate_angle_comment(row: pd.Series) -> str:
    angle_name = row["largest_angle_difference"]
    diff_value = row["largest_angle_mean_abs_diff_deg"]
    label = ANGLE_LABELS.get(angle_name, angle_name)
    group = get_angle_group(angle_name)
    suggestion = FEEDBACK_RULES.get(
        group,
        "目標と形が異なる区間です。該当時間のポーズを見比べてください。",
    )

    return f"{label}の差が大きい区間です。平均角度差は{diff_value:.1f}°です。提案: {suggestion}"


def create_angle_events(
    summary_df: pd.DataFrame,
    angle_diff_threshold_deg: float = ANGLE_DIFF_THRESHOLD_DEG,
) -> pd.DataFrame:
    event_df = summary_df[
        summary_df["largest_angle_mean_abs_diff_deg"] >= angle_diff_threshold_deg
    ].copy()

    if event_df.empty:
        return event_df

    event_df["comment"] = event_df.apply(generate_angle_comment, axis=1)
    event_df = event_df.sort_values("largest_angle_mean_abs_diff_deg", ascending=False)
    return event_df


def plot_angle_difference_timeseries(
    summary_df: pd.DataFrame,
    output_path: Path,
    angle_diff_threshold_deg: float = ANGLE_DIFF_THRESHOLD_DEG,
) -> Path:
    plt.figure(figsize=(12, 6))

    x = (summary_df["start_sec"] + summary_df["end_sec"]) / 2
    y = summary_df["largest_angle_mean_abs_diff_deg"]

    plt.plot(x, y, marker="o", linewidth=1.5, label="largest angle difference")
    plt.axhline(angle_diff_threshold_deg, linestyle="--", label="threshold")
    plt.xlabel("time [s]")
    plt.ylabel("mean absolute angle difference [deg]")
    plt.title("Pose angle difference between target and learner")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def create_text_report(
    summary_df: pd.DataFrame,
    event_df: pd.DataFrame,
    output_path: Path,
    window_sec: float = WINDOW_SEC,
    angle_diff_threshold_deg: float = ANGLE_DIFF_THRESHOLD_DEG,
) -> Path:
    lines = []
    lines.append("Pose Angle Comparison Report")
    lines.append("=" * 50)
    lines.append("person_0: target dancer")
    lines.append("person_1: learner dancer")
    lines.append(f"window_sec: {window_sec}")
    lines.append(f"event_threshold: {angle_diff_threshold_deg} deg")
    lines.append("")

    lines.append("[角度差が大きい上位区間]")
    top_rows = summary_df.sort_values("largest_angle_mean_abs_diff_deg", ascending=False).head(10)
    for _, row in top_rows.iterrows():
        angle_name = row["largest_angle_difference"]
        label = ANGLE_LABELS.get(angle_name, angle_name)
        lines.append(
            f"- {row['start_sec']:.1f}-{row['end_sec']:.1f}s: "
            f"{label}, mean_abs_diff={row['largest_angle_mean_abs_diff_deg']:.1f}°"
        )

    lines.append("")
    lines.append("[閾値を超えたイベント]")
    if event_df.empty:
        lines.append("- 閾値を超える角度差イベントは検出されませんでした。")
    else:
        for _, row in event_df.iterrows():
            lines.append(f"- {row['start_sec']:.1f}-{row['end_sec']:.1f}s: {row['comment']}")

    lines.append("")
    lines.append("[指標の意味]")
    lines.append("- elbow_angle: 肘の曲げ伸ばし。腕が伸びているか、曲がっているかを示します。")
    lines.append("- knee_angle: 膝の曲げ伸ばし。沈み込みやステップ時の膝の使い方を示します。")
    lines.append("- shoulder_angle: 肩から腕を出す方向。腕の形や上半身との関係を示します。")
    lines.append("- hip_angle: 股関節まわりの形。上半身と脚の角度、体重の乗せ方に関係します。")
    lines.append("- torso_lean_angle: 胴体の傾き。上半身が左右どちらに倒れているかを示します。")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_angle_comparison(
    target_csv_path: Path = TARGET_CSV_PATH,
    learner_csv_path: Path = LEARNER_CSV_PATH,
    output_dir: Path = OUTPUT_DIR,
    window_sec: float = WINDOW_SEC,
    angle_diff_threshold_deg: float = ANGLE_DIFF_THRESHOLD_DEG,
    min_valid_frames_per_window: int = MIN_VALID_FRAMES_PER_WINDOW,
    decimals: int = 3,
    create_plot: bool = True,
) -> Dict[str, Path]:
    """Compare pose angles between target dancer and learner dancer.

    Designed to be called from a Web app such as Streamlit.
    Returns paths to generated outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    target_df = load_pose_csv(target_csv_path)
    learner_df = load_pose_csv(learner_csv_path)

    target_angle_df = add_angle_columns(target_df)
    learner_angle_df = add_angle_columns(learner_df)
    merged_df = merge_target_and_learner(target_angle_df, learner_angle_df)

    summary_df = summarize_angle_differences_by_window(
        merged_df,
        window_sec=window_sec,
        min_valid_frames_per_window=min_valid_frames_per_window,
    )
    if summary_df.empty:
        raise RuntimeError("No valid angle comparison windows were found.")

    event_df = create_angle_events(
        summary_df,
        angle_diff_threshold_deg=angle_diff_threshold_deg,
    )

    summary_csv = output_dir / "angle_difference_summary.csv"
    events_csv = output_dir / "angle_difference_events.csv"
    plot_path = output_dir / "angle_difference_timeseries.png"
    report_path = output_dir / "angle_difference_report.txt"

    summary_df.round(decimals).to_csv(summary_csv, index=False)
    event_df.round(decimals).to_csv(events_csv, index=False)

    outputs: Dict[str, Path] = {
        "angle_difference_summary_csv": summary_csv,
        "angle_difference_events_csv": events_csv,
        "angle_difference_report_txt": create_text_report(
            summary_df,
            event_df,
            report_path,
            window_sec=window_sec,
            angle_diff_threshold_deg=angle_diff_threshold_deg,
        ),
    }

    if create_plot:
        outputs["angle_difference_plot"] = plot_angle_difference_timeseries(
            summary_df,
            plot_path,
            angle_diff_threshold_deg=angle_diff_threshold_deg,
        )

    for output_path in outputs.values():
        print(f"Saved: {output_path}")

    return outputs



def main() -> None:
    run_angle_comparison(
        target_csv_path=TARGET_CSV_PATH,
        learner_csv_path=LEARNER_CSV_PATH,
        output_dir=OUTPUT_DIR,
        window_sec=WINDOW_SEC,
        angle_diff_threshold_deg=ANGLE_DIFF_THRESHOLD_DEG,
        min_valid_frames_per_window=MIN_VALID_FRAMES_PER_WINDOW,
        decimals=3,
        create_plot=True,
    )


if __name__ == "__main__":
    main()