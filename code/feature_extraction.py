"""
Feature extraction utilities for pose time-series CSV files.

This module can be used in two ways:
1. As a script:
    python code/feature_extraction.py

2. As a library:
    from feature_extraction import run_feature_extraction

    outputs = run_feature_extraction(
        input_csv_path=Path("outputs/multi/person_0_pose.csv"),
        output_dir=Path("outputs/multi/features_0"),
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV_PATH = Path("outputs/multi/person_1_pose.csv")
OUTPUT_DIR = Path("outputs/multi/features_1")

TRAJECTORY_JOINTS = [
    "left_wrist",
    "right_wrist",
    "left_ankle",
    "right_ankle",
]

WRIST_JOINTS = [
    "left_wrist",
    "right_wrist",
]

FEATURE_JOINTS = [
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

MOVEMENT_SIZE_JOINTS = [
    "left_wrist",
    "right_wrist",
    "left_ankle",
    "right_ankle",
]


def load_pose_csv(csv_path: Path) -> pd.DataFrame:
    """Load pose CSV and convert numeric columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    for column in df.columns:
        if column != "pose_detected":
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["pose_detected"] = pd.to_numeric(df["pose_detected"], errors="coerce").fillna(0).astype(int)
    return df


def get_xy(df: pd.DataFrame, joint_name: str) -> Tuple[pd.Series, pd.Series]:
    """Return x and y series for a joint."""
    return df[f"{joint_name}_x"], df[f"{joint_name}_y"]


def normalize_pose_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize coordinates using hip center as origin and shoulder width as scale."""
    normalized_df = df.copy()

    hip_center_x = (df["left_hip_x"] + df["right_hip_x"]) / 2
    hip_center_y = (df["left_hip_y"] + df["right_hip_y"]) / 2

    shoulder_width = np.sqrt(
        (df["left_shoulder_x"] - df["right_shoulder_x"]) ** 2
        + (df["left_shoulder_y"] - df["right_shoulder_y"]) ** 2
    )

    shoulder_width = shoulder_width.replace(0, np.nan)

    for column in df.columns:
        if column.endswith("_x"):
            normalized_df[column] = (df[column] - hip_center_x) / shoulder_width

        elif column.endswith("_y"):
            normalized_df[column] = (df[column] - hip_center_y) / shoulder_width

    return normalized_df


def plot_2d_trajectories(df: pd.DataFrame, joints: Iterable[str], output_path: Path) -> None:
    """Plot 2D trajectories of selected joints."""
    detected_df = df[df["pose_detected"] == 1].copy()

    plt.figure(figsize=(8, 8))

    for joint in joints:
        x, y = get_xy(detected_df, joint)
        valid = x.notna() & y.notna()
        plt.plot(x[valid], y[valid], linewidth=1.5, label=joint)
        if valid.any():
            plt.scatter(x[valid].iloc[0], y[valid].iloc[0], s=30, marker="o")
            plt.scatter(x[valid].iloc[-1], y[valid].iloc[-1], s=30, marker="x")

    plt.gca().invert_yaxis()
    plt.xlabel("normalized x (hip-centered)")
    plt.ylabel("normalized y (hip-centered)")
    plt.title("2D trajectories of wrists and ankles")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_joint_speed(df: pd.DataFrame, joint_name: str) -> pd.Series:
    """Compute frame-wise 2D speed for one joint using normalized image coordinates."""
    x, y = get_xy(df, joint_name)
    t = df["timestamp_sec"]

    dx = x.diff()
    dy = y.diff()
    dt = t.diff()

    speed = np.sqrt(dx**2 + dy**2) / dt
    speed[(df["pose_detected"] == 0) | (df["pose_detected"].shift(1) == 0)] = np.nan
    speed[dt <= 0] = np.nan
    return speed


# ---- Feature summary and export ----
def compute_total_distance(df: pd.DataFrame, joint_name: str) -> float:
    x, y = get_xy(df, joint_name)
    dx = x.diff()
    dy = y.diff()
    distance = np.sqrt(dx**2 + dy**2)
    return float(distance.sum(skipna=True))


def compute_joint_acceleration(df: pd.DataFrame, joint_name: str) -> pd.Series:
    speed = compute_joint_speed(df, joint_name)
    dt = df["timestamp_sec"].diff()
    acceleration = speed.diff() / dt
    acceleration[dt <= 0] = np.nan
    return acceleration


def compute_joint_jerk(df: pd.DataFrame, joint_name: str) -> pd.Series:
    acceleration = compute_joint_acceleration(df, joint_name)
    dt = df["timestamp_sec"].diff()
    jerk = acceleration.diff() / dt
    jerk[dt <= 0] = np.nan
    return jerk


def calculate_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = {}

    stop_threshold = 0.5

    for joint in FEATURE_JOINTS:
        speed = compute_joint_speed(df, joint)
        jerk = compute_joint_jerk(df, joint)

        summary[f"{joint}_total_distance"] = compute_total_distance(df, joint)
        summary[f"{joint}_mean_speed"] = float(speed.mean(skipna=True))
        summary[f"{joint}_max_speed"] = float(speed.max(skipna=True))
        summary[f"{joint}_speed_std"] = float(speed.std(skipna=True))
        summary[f"{joint}_mean_abs_jerk"] = float(np.abs(jerk).mean(skipna=True))
        summary[f"{joint}_max_abs_jerk"] = float(np.abs(jerk).max(skipna=True))

    right_speed = compute_joint_speed(df, "right_wrist")
    summary["stop_ratio"] = float((right_speed < stop_threshold).mean())

    summary["movement_size_score"] = sum(
        summary[f"{joint}_total_distance"] for joint in MOVEMENT_SIZE_JOINTS
    )

    summary["dynamic_score"] = float(
        np.mean([summary[f"{joint}_speed_std"] for joint in MOVEMENT_SIZE_JOINTS])
    )

    summary["smoothness_score"] = float(
        np.mean([summary[f"{joint}_mean_abs_jerk"] for joint in MOVEMENT_SIZE_JOINTS])
    )

    # Add x/y range for each FEATURE_JOINTS
    for joint in FEATURE_JOINTS:
        x, y = get_xy(df, joint)
        valid = x.notna() & y.notna()
        summary[f"{joint}_x_range"] = float(x[valid].max() - x[valid].min()) if valid.any() else np.nan
        summary[f"{joint}_y_range"] = float(y[valid].max() - y[valid].min()) if valid.any() else np.nan

    return pd.DataFrame([summary])


def calculate_window_feature_summary(
    df: pd.DataFrame,
    window_sec: float = 2.0,
) -> pd.DataFrame:
    """Calculate feature summaries for fixed-length time windows."""

    max_time = float(df["timestamp_sec"].max())
    summaries = []

    window_id = 0
    start_time = 0.0

    while start_time < max_time:
        end_time = start_time + window_sec

        window_df = df[
            (df["timestamp_sec"] >= start_time)
            & (df["timestamp_sec"] < end_time)
        ].copy()

        if len(window_df) < 5:
            start_time = end_time
            window_id += 1
            continue

        feature_df = calculate_feature_summary(window_df)
        feature_dict = feature_df.iloc[0].to_dict()

        feature_dict["window_id"] = window_id
        feature_dict["start_sec"] = start_time
        feature_dict["end_sec"] = end_time

        summaries.append(feature_dict)

        start_time = end_time
        window_id += 1

    return pd.DataFrame(summaries)


# ---- DataFrame helpers ----
def reorder_columns(df: pd.DataFrame, first_columns: list[str]) -> pd.DataFrame:
    """Move selected columns to the beginning of the dataframe."""
    existing_first_columns = [column for column in first_columns if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in existing_first_columns]
    return df[existing_first_columns + remaining_columns]


def round_numeric_columns(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Round numeric columns for easier inspection in CSV files."""
    rounded_df = df.copy()
    numeric_columns = rounded_df.select_dtypes(include=[np.number]).columns
    rounded_df[numeric_columns] = rounded_df[numeric_columns].round(decimals)
    return rounded_df


def plot_wrist_speed(df: pd.DataFrame, output_path: Path) -> None:
    """Plot left and right wrist speed time-series."""
    plt.figure(figsize=(10, 5))

    for joint in WRIST_JOINTS:
        speed = compute_joint_speed(df, joint)
        plt.plot(df["timestamp_sec"], speed, linewidth=1.2, label=f"{joint}_speed")

    plt.xlabel("time [s]")
    plt.ylabel("normalized speed [body scale / s]")
    plt.title("Wrist speed time-series")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pose_detected(df: pd.DataFrame, output_path: Path) -> None:
    """Plot pose detection status over time."""
    plt.figure(figsize=(10, 3))
    plt.step(df["timestamp_sec"], df["pose_detected"], where="post")
    plt.xlabel("time [s]")
    plt.ylabel("pose_detected")
    plt.yticks([0, 1])
    plt.ylim(-0.1, 1.1)
    plt.title("Pose detection time-series")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_feature_extraction(
    input_csv_path: Path,
    output_dir: Path,
    window_sec: float = 2.0,
    create_plots: bool = True,
    decimals: int = 3,
) -> Dict[str, Path]:
    """Run all feature extraction steps for one pose CSV file.

    This function is designed to be called from a Web app such as Streamlit.
    It returns paths to generated outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_pose_csv(input_csv_path)

    # Body-size normalization
    # Origin: hip center
    # Scale: shoulder width
    normalized_df = normalize_pose_coordinates(df)

    trajectory_path = output_dir / "trajectory_wrists_ankles.png"
    wrist_speed_path = output_dir / "wrist_speed_timeseries.png"
    pose_detected_path = output_dir / "pose_detected_timeseries.png"
    feature_summary_path = output_dir / "feature_summary.csv"
    window_feature_summary_path = output_dir / "window_feature_summary.csv"

    outputs: Dict[str, Path] = {
        "feature_summary_csv": feature_summary_path,
        "window_feature_summary_csv": window_feature_summary_path,
    }

    if create_plots:
        plot_2d_trajectories(normalized_df, TRAJECTORY_JOINTS, trajectory_path)
        plot_wrist_speed(normalized_df, wrist_speed_path)
        plot_pose_detected(df, pose_detected_path)
        outputs["trajectory_plot"] = trajectory_path
        outputs["wrist_speed_plot"] = wrist_speed_path
        outputs["pose_detected_plot"] = pose_detected_path

    feature_summary = calculate_feature_summary(normalized_df)
    feature_summary = round_numeric_columns(feature_summary, decimals=decimals)
    feature_summary.to_csv(feature_summary_path, index=False)

    window_feature_summary = calculate_window_feature_summary(
        normalized_df,
        window_sec=window_sec,
    )
    window_feature_summary = reorder_columns(
        window_feature_summary,
        first_columns=["window_id", "start_sec", "end_sec"],
    )
    window_feature_summary = round_numeric_columns(window_feature_summary, decimals=decimals)
    window_feature_summary.to_csv(window_feature_summary_path, index=False)

    for output_path in outputs.values():
        print(f"Saved: {output_path}")

    return outputs


def main() -> None:
    run_feature_extraction(
        input_csv_path=INPUT_CSV_PATH,
        output_dir=OUTPUT_DIR,
        window_sec=2.0,
        create_plots=True,
        decimals=3,
    )


if __name__ == "__main__":
    main()