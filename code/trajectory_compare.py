

"""
Compare joint trajectories between target and learner in a specified time range.

Inputs:
    outputs/multi/person_0_pose.csv  # target dancer
    outputs/multi/person_1_pose.csv  # learner dancer

Outputs:
    outputs/multi/trajectory_compare/trajectory_<joint>_<start>_<end>.png
    outputs/multi/trajectory_compare/trajectory_<joint>_<start>_<end>.csv

Run:
    python code/trajectory_compare.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Literal, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_CSV_PATH = Path("outputs/multi/person_0_pose.csv")
LEARNER_CSV_PATH = Path("outputs/multi/person_1_pose.csv")
VIDEO_PATH = Path("outputs/multi/annotated_video.mp4")
OUTPUT_DIR = Path("outputs/multi/trajectory_compare")

START_SEC = 0.0
END_SEC = 5.0
JOINT_NAME = "right_wrist"
VISIBILITY_THRESHOLD = 0.5
BACKGROUND_TIME_MODE: Literal["start", "end"] = "start"
def prepare_video_overlay_dataframe(
    target_df: pd.DataFrame,
    learner_df: pd.DataFrame,
    joint_name: str,
    start_sec: float,
    end_sec: float,
) -> pd.DataFrame:
    """Prepare raw video-coordinate trajectories for drawing on the original frame."""
    required_columns = [
        "frame_index",
        "timestamp_sec",
        "pose_detected",
        f"{joint_name}_x",
        f"{joint_name}_y",
        f"{joint_name}_visibility",
    ]
    for name, df in [("target", target_df), ("learner", learner_df)]:
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing columns in {name} CSV: {missing_columns}")

    target_range_df = filter_time_range(target_df, start_sec, end_sec)
    learner_range_df = filter_time_range(learner_df, start_sec, end_sec)

    target_valid_df = target_range_df[
        target_range_df[f"{joint_name}_visibility"] >= VISIBILITY_THRESHOLD
    ][["frame_index", "timestamp_sec", f"{joint_name}_x", f"{joint_name}_y"]].rename(
        columns={
            f"{joint_name}_x": "target_x",
            f"{joint_name}_y": "target_y",
        }
    )
    learner_valid_df = learner_range_df[
        learner_range_df[f"{joint_name}_visibility"] >= VISIBILITY_THRESHOLD
    ][["frame_index", "timestamp_sec", f"{joint_name}_x", f"{joint_name}_y"]].rename(
        columns={
            f"{joint_name}_x": "learner_x",
            f"{joint_name}_y": "learner_y",
        }
    )

    merged_df = pd.merge(
        target_valid_df,
        learner_valid_df,
        on=["frame_index", "timestamp_sec"],
        how="inner",
    )

    if merged_df.empty:
        return merged_df

    merged_df["trajectory_distance_px"] = np.sqrt(
        (merged_df["learner_x"] - merged_df["target_x"]) ** 2
        + (merged_df["learner_y"] - merged_df["target_y"]) ** 2
    )
    merged_df["time_from_start_sec"] = merged_df["timestamp_sec"] - start_sec
    return merged_df


def read_video_frame(video_path: Path, timestamp_sec: float) -> Tuple[np.ndarray, int]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Failed to read FPS from video: {video_path}")

    frame_index = max(0, int(round(timestamp_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame_bgr = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(
            f"Failed to read frame at {timestamp_sec:.2f}s from video: {video_path}"
        )

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb, frame_index


def plot_trajectory_on_video_frame(
    overlay_df: pd.DataFrame,
    video_path: Path,
    output_path: Path,
    joint_name: str,
    start_sec: float,
    end_sec: float,
    background_time_mode: Literal["start", "end"] = "start",
) -> Path:
    """Draw target/learner trajectories on the original video frame.

    The default background is the start frame because it makes it easier to compare
    how each dancer moves away from the same initial timing.
    """
    if background_time_mode not in {"start", "end"}:
        raise ValueError("background_time_mode must be 'start' or 'end'.")

    background_time_sec = start_sec if background_time_mode == "start" else end_sec
    frame_rgb, background_frame_index = read_video_frame(video_path, background_time_sec)
    joint_label = JOINT_LABELS.get(joint_name, joint_name)

    height, width = frame_rgb.shape[:2]

    # MediaPipe coordinates are often normalized to 0-1. Convert them to pixel
    # coordinates before drawing on the video frame. If the coordinates are already
    # pixel values, keep them unchanged.
    draw_df = overlay_df.copy()
    coordinate_columns = ["target_x", "target_y", "learner_x", "learner_y"]
    if not draw_df.empty:
        max_abs_coordinate = draw_df[coordinate_columns].abs().max().max()
        if max_abs_coordinate <= 2.0:
            draw_df["target_x"] = draw_df["target_x"] * width
            draw_df["learner_x"] = draw_df["learner_x"] * width
            draw_df["target_y"] = draw_df["target_y"] * height
            draw_df["learner_y"] = draw_df["learner_y"] * height

        draw_df["trajectory_distance_px"] = np.sqrt(
            (draw_df["learner_x"] - draw_df["target_x"]) ** 2
            + (draw_df["learner_y"] - draw_df["target_y"]) ** 2
        )

    plt.figure(figsize=(12, 7))
    plt.imshow(frame_rgb)

    if draw_df.empty:
        plt.text(
            width / 2,
            height / 2,
            "No valid trajectory data",
            ha="center",
            va="center",
            color="white",
            fontsize=16,
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 8},
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()
        return output_path

    plt.plot(
        draw_df["target_x"],
        draw_df["target_y"],
        color="lime",
        marker="o",
        markersize=3,
        linewidth=2,
        label="target trajectory",
    )
    plt.plot(
        draw_df["learner_x"],
        draw_df["learner_y"],
        color="red",
        marker="o",
        markersize=3,
        linewidth=2,
        label="learner trajectory",
    )

    # Highlight frames where the video-coordinate trajectory difference is large.
    high_diff_threshold = draw_df["trajectory_distance_px"].quantile(0.8)
    high_diff_df = draw_df[draw_df["trajectory_distance_px"] >= high_diff_threshold]
    for _, row in high_diff_df.iterrows():
        plt.plot(
            [row["target_x"], row["learner_x"]],
            [row["target_y"], row["learner_y"]],
            color="yellow",
            linewidth=1.2,
            alpha=0.8,
        )

    # Start and end markers.
    plt.scatter(
        draw_df.iloc[0]["target_x"],
        draw_df.iloc[0]["target_y"],
        marker="o",
        s=80,
        facecolors="none",
        edgecolors="white",
        linewidths=2,
        label="start",
    )
    plt.scatter(
        draw_df.iloc[-1]["target_x"],
        draw_df.iloc[-1]["target_y"],
        marker="x",
        s=90,
        color="white",
        linewidths=2,
        label="end",
    )
    plt.scatter(
        draw_df.iloc[0]["learner_x"],
        draw_df.iloc[0]["learner_y"],
        marker="o",
        s=80,
        facecolors="none",
        edgecolors="white",
        linewidths=2,
    )
    plt.scatter(
        draw_df.iloc[-1]["learner_x"],
        draw_df.iloc[-1]["learner_y"],
        marker="x",
        s=90,
        color="white",
        linewidths=2,
    )

    mean_diff_px = draw_df["trajectory_distance_px"].mean()
    max_diff_px = draw_df["trajectory_distance_px"].max()
    target_length_px = calculate_total_path_length(
        draw_df[["target_x", "target_y"]].to_numpy()
    )
    learner_length_px = calculate_total_path_length(
        draw_df[["learner_x", "learner_y"]].to_numpy()
    )

    title_text = (
        f"Trajectory on video: {joint_label} | "
        f"{start_sec:.1f}-{end_sec:.1f}s | "
        f"background={background_time_mode} frame ({background_frame_index})"
    )
    summary_text = (
        f"mean diff={mean_diff_px:.1f}px, max diff={max_diff_px:.1f}px, "
        f"target length={target_length_px:.1f}px, learner length={learner_length_px:.1f}px"
    )
    plt.text(
        12,
        24,
        title_text,
        color="white",
        fontsize=12,
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 6},
    )
    plt.text(
        12,
        height - 20,
        summary_text,
        color="white",
        fontsize=11,
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 6},
    )
    plt.legend(loc="upper right", fontsize=9)
    plt.xlim(0, width)
    plt.ylim(height, 0)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
    return output_path

AVAILABLE_JOINTS = [
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

JOINT_LABELS = {
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


def load_pose_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for column in df.columns:
        if column != "pose_detected":
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["pose_detected"] = pd.to_numeric(df["pose_detected"], errors="coerce").fillna(0).astype(int)
    return df


def validate_joint_name(joint_name: str, available_joints: Iterable[str] = AVAILABLE_JOINTS) -> None:
    if joint_name not in available_joints:
        raise ValueError(
            f"Invalid joint_name: {joint_name}. "
            f"Choose from: {', '.join(available_joints)}"
        )


def filter_time_range(df: pd.DataFrame, start_sec: float, end_sec: float) -> pd.DataFrame:
    if end_sec <= start_sec:
        raise ValueError("end_sec must be larger than start_sec.")

    return df[
        (df["timestamp_sec"] >= start_sec)
        & (df["timestamp_sec"] <= end_sec)
        & (df["pose_detected"] == 1)
    ].copy()


def add_normalized_coordinates(df: pd.DataFrame, joint_name: str) -> pd.DataFrame:
    """Normalize joint coordinates by hip center and shoulder width.

    This removes translation and body-size differences so trajectories can be compared
    as form differences rather than position differences in the video.
    """
    required_columns = [
        f"{joint_name}_x",
        f"{joint_name}_y",
        f"{joint_name}_visibility",
        "left_hip_x",
        "left_hip_y",
        "right_hip_x",
        "right_hip_y",
        "left_shoulder_x",
        "left_shoulder_y",
        "right_shoulder_x",
        "right_shoulder_y",
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}")

    result_df = df.copy()

    joint_visibility = result_df[f"{joint_name}_visibility"]
    valid_visibility = joint_visibility >= VISIBILITY_THRESHOLD

    hip_center_x = (result_df["left_hip_x"] + result_df["right_hip_x"]) / 2
    hip_center_y = (result_df["left_hip_y"] + result_df["right_hip_y"]) / 2

    shoulder_width = np.sqrt(
        (result_df["left_shoulder_x"] - result_df["right_shoulder_x"]) ** 2
        + (result_df["left_shoulder_y"] - result_df["right_shoulder_y"]) ** 2
    )
    valid_scale = shoulder_width > 1e-6

    valid_rows = valid_visibility & valid_scale

    result_df["joint_x_norm"] = np.nan
    result_df["joint_y_norm"] = np.nan
    result_df.loc[valid_rows, "joint_x_norm"] = (
        result_df.loc[valid_rows, f"{joint_name}_x"] - hip_center_x.loc[valid_rows]
    ) / shoulder_width.loc[valid_rows]
    result_df.loc[valid_rows, "joint_y_norm"] = (
        result_df.loc[valid_rows, f"{joint_name}_y"] - hip_center_y.loc[valid_rows]
    ) / shoulder_width.loc[valid_rows]

    result_df = result_df.dropna(subset=["joint_x_norm", "joint_y_norm"]).copy()
    return result_df


def prepare_trajectory_dataframe(
    target_df: pd.DataFrame,
    learner_df: pd.DataFrame,
    joint_name: str,
    start_sec: float,
    end_sec: float,
) -> pd.DataFrame:
    target_range_df = filter_time_range(target_df, start_sec, end_sec)
    learner_range_df = filter_time_range(learner_df, start_sec, end_sec)

    target_norm_df = add_normalized_coordinates(target_range_df, joint_name)
    learner_norm_df = add_normalized_coordinates(learner_range_df, joint_name)

    target_part = target_norm_df[
        ["frame_index", "timestamp_sec", "joint_x_norm", "joint_y_norm"]
    ].rename(
        columns={
            "joint_x_norm": "target_x_norm",
            "joint_y_norm": "target_y_norm",
        }
    )
    learner_part = learner_norm_df[
        ["frame_index", "timestamp_sec", "joint_x_norm", "joint_y_norm"]
    ].rename(
        columns={
            "joint_x_norm": "learner_x_norm",
            "joint_y_norm": "learner_y_norm",
        }
    )

    merged_df = pd.merge(
        target_part,
        learner_part,
        on=["frame_index", "timestamp_sec"],
        how="inner",
    )

    if merged_df.empty:
        return merged_df

    merged_df["trajectory_distance"] = np.sqrt(
        (merged_df["learner_x_norm"] - merged_df["target_x_norm"]) ** 2
        + (merged_df["learner_y_norm"] - merged_df["target_y_norm"]) ** 2
    )
    merged_df["time_from_start_sec"] = merged_df["timestamp_sec"] - start_sec

    return merged_df


def summarize_trajectory_difference(trajectory_df: pd.DataFrame) -> Dict[str, float]:
    if trajectory_df.empty:
        return {
            "mean_distance": np.nan,
            "max_distance": np.nan,
            "target_total_distance": np.nan,
            "learner_total_distance": np.nan,
            "distance_ratio": np.nan,
        }

    target_points = trajectory_df[["target_x_norm", "target_y_norm"]].to_numpy()
    learner_points = trajectory_df[["learner_x_norm", "learner_y_norm"]].to_numpy()

    target_total_distance = calculate_total_path_length(target_points)
    learner_total_distance = calculate_total_path_length(learner_points)

    if target_total_distance <= 1e-6:
        distance_ratio = np.nan
    else:
        distance_ratio = learner_total_distance / target_total_distance

    return {
        "mean_distance": float(trajectory_df["trajectory_distance"].mean()),
        "max_distance": float(trajectory_df["trajectory_distance"].max()),
        "target_total_distance": float(target_total_distance),
        "learner_total_distance": float(learner_total_distance),
        "distance_ratio": float(distance_ratio) if not pd.isna(distance_ratio) else np.nan,
    }


def calculate_total_path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0

    diff = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diff, axis=1)
    return float(segment_lengths.sum())


def plot_trajectory_comparison(
    trajectory_df: pd.DataFrame,
    output_path: Path,
    joint_name: str,
    start_sec: float,
    end_sec: float,
) -> Path:
    joint_label = JOINT_LABELS.get(joint_name, joint_name)

    plt.figure(figsize=(7, 7))

    if trajectory_df.empty:
        plt.text(0.5, 0.5, "No valid trajectory data", ha="center", va="center")
        plt.axis("off")
        plt.savefig(output_path, dpi=200)
        plt.close()
        return output_path

    plt.plot(
        trajectory_df["target_x_norm"],
        -trajectory_df["target_y_norm"],
        marker="o",
        markersize=3,
        linewidth=2,
        label="target",
    )
    plt.plot(
        trajectory_df["learner_x_norm"],
        -trajectory_df["learner_y_norm"],
        marker="o",
        markersize=3,
        linewidth=2,
        label="learner",
    )

    # Mark start and end points.
    plt.scatter(
        trajectory_df.iloc[0]["target_x_norm"],
        -trajectory_df.iloc[0]["target_y_norm"],
        marker="s",
        s=80,
        label="target start",
    )
    plt.scatter(
        trajectory_df.iloc[-1]["target_x_norm"],
        -trajectory_df.iloc[-1]["target_y_norm"],
        marker="X",
        s=80,
        label="target end",
    )
    plt.scatter(
        trajectory_df.iloc[0]["learner_x_norm"],
        -trajectory_df.iloc[0]["learner_y_norm"],
        marker="s",
        s=80,
        label="learner start",
    )
    plt.scatter(
        trajectory_df.iloc[-1]["learner_x_norm"],
        -trajectory_df.iloc[-1]["learner_y_norm"],
        marker="X",
        s=80,
        label="learner end",
    )

    summary = summarize_trajectory_difference(trajectory_df)
    title = f"Trajectory comparison: {joint_label} ({start_sec:.1f}-{end_sec:.1f}s)"
    subtitle = (
        f"mean diff={summary['mean_distance']:.3f}, "
        f"target dist={summary['target_total_distance']:.3f}, "
        f"learner dist={summary['learner_total_distance']:.3f}"
    )

    plt.title(f"{title}\n{subtitle}")
    plt.xlabel("normalized x")
    plt.ylabel("normalized y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def create_output_file_stem(joint_name: str, start_sec: float, end_sec: float) -> str:
    safe_start = f"{start_sec:.1f}".replace(".", "_")
    safe_end = f"{end_sec:.1f}".replace(".", "_")
    return f"trajectory_{joint_name}_{safe_start}_{safe_end}"


def run_trajectory_compare(
    target_csv_path: Path = TARGET_CSV_PATH,
    learner_csv_path: Path = LEARNER_CSV_PATH,
    video_path: Path = VIDEO_PATH,
    output_dir: Path = OUTPUT_DIR,
    start_sec: float = START_SEC,
    end_sec: float = END_SEC,
    joint_name: str = JOINT_NAME,
    decimals: int = 3,
    create_plot: bool = True,
    create_video_overlay: bool = True,
    background_time_mode: Literal["start", "end"] = BACKGROUND_TIME_MODE,
) -> Dict[str, Path]:
    """Compare target and learner trajectories for one joint and time range.

    Outputs both a normalized coordinate plot and, when requested, an overlay
    image that draws the raw joint trajectory directly on the original video.
    Designed to be called from a Web app such as Streamlit.
    """
    validate_joint_name(joint_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_df = load_pose_csv(target_csv_path)
    learner_df = load_pose_csv(learner_csv_path)

    trajectory_df = prepare_trajectory_dataframe(
        target_df=target_df,
        learner_df=learner_df,
        joint_name=joint_name,
        start_sec=start_sec,
        end_sec=end_sec,
    )

    file_stem = create_output_file_stem(joint_name, start_sec, end_sec)
    trajectory_csv_path = output_dir / f"{file_stem}.csv"
    trajectory_plot_path = output_dir / f"{file_stem}.png"
    trajectory_video_overlay_path = output_dir / f"{file_stem}_video_overlay_{background_time_mode}.png"

    summary = summarize_trajectory_difference(trajectory_df)
    for key, value in summary.items():
        trajectory_df[key] = value

    trajectory_df.round(decimals).to_csv(trajectory_csv_path, index=False)

    outputs: Dict[str, Path] = {
        "trajectory_compare_csv": trajectory_csv_path,
    }

    if create_plot:
        outputs["trajectory_compare_plot"] = plot_trajectory_comparison(
            trajectory_df=trajectory_df,
            output_path=trajectory_plot_path,
            joint_name=joint_name,
            start_sec=start_sec,
            end_sec=end_sec,
        )

    if create_video_overlay:
        overlay_df = prepare_video_overlay_dataframe(
            target_df=target_df,
            learner_df=learner_df,
            joint_name=joint_name,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        outputs["trajectory_video_overlay_plot"] = plot_trajectory_on_video_frame(
            overlay_df=overlay_df,
            video_path=video_path,
            output_path=trajectory_video_overlay_path,
            joint_name=joint_name,
            start_sec=start_sec,
            end_sec=end_sec,
            background_time_mode=background_time_mode,
        )

    for output_path in outputs.values():
        print(f"Saved: {output_path}")

    return outputs


def main() -> None:
    run_trajectory_compare(
        target_csv_path=TARGET_CSV_PATH,
        learner_csv_path=LEARNER_CSV_PATH,
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        start_sec=START_SEC,
        end_sec=END_SEC,
        joint_name=JOINT_NAME,
        decimals=3,
        create_plot=True,
        create_video_overlay=True,
        background_time_mode=BACKGROUND_TIME_MODE,
    )


if __name__ == "__main__":
    main()