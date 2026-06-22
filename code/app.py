

"""
Streamlit app for dance motion comparison.

Run:
    streamlit run code/app.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st


# Allow importing sibling modules when this app is launched from project root.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from compare_dancers import run_dancer_comparison
from data_aquisition_multi import run_multi_person_pose_extraction
from difference_timeline import run_difference_timeline
from feature_extraction_multi import run_multi_feature_extraction
from pose_angle_comparison import run_angle_comparison
from pose_snapshot import run_pose_snapshot
from trajectory_dynamics_overlay import create_trajectory_dynamics_overlay_video
from video_overlay import create_angle_overlay_video


APP_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "app_runs"
POSE_MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
YOLO_MODEL_PATH = "yolo11n.pt"


st.set_page_config(
    page_title="Dance Motion Comparison",
    layout="wide",
)


def create_run_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = APP_OUTPUT_ROOT / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_uploaded_video(uploaded_file, run_dir: Path) -> Path:
    suffix = Path(uploaded_file.name).suffix
    if not suffix:
        suffix = ".mp4"

    input_video_path = run_dir / f"input_video{suffix}"
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return input_video_path


def read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


# Helper to convert generated video to H.264 mp4 for Streamlit playback
def prepare_video_for_streamlit(video_path: Path) -> Path:
    """Convert a generated video to a browser-friendly H.264 mp4 when possible."""
    if not video_path.exists():
        return video_path

    web_video_path = video_path.with_name(f"{video_path.stem}_streamlit.mp4")
    if web_video_path.exists() and web_video_path.stat().st_size > 0:
        return web_video_path

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(web_video_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if web_video_path.exists() and web_video_path.stat().st_size > 0:
            return web_video_path
    except Exception:
        return video_path

    return video_path


def show_csv_preview(path: Path, title: str, max_rows: int = 10) -> None:
    if not path.exists():
        st.warning(f"{title} が見つかりません: {path}")
        return

    st.subheader(title)
    df = pd.read_csv(path)
    st.dataframe(df.head(max_rows), use_container_width=True)


def show_video(path: Path, title: str) -> None:
    if not path.exists():
        st.warning(f"{title} が見つかりません: {path}")
        return

    st.subheader(title)
    display_path = prepare_video_for_streamlit(path)

    try:
        st.video(display_path.read_bytes())
    except Exception:
        st.video(str(display_path))

    with open(display_path, "rb") as video_file:
        st.download_button(
            label=f"{title} をダウンロード",
            data=video_file,
            file_name=display_path.name,
            mime="video/mp4",
        )


# Show top feedback cards helper
def show_top_feedback_cards(path: Path, max_items: int = 5) -> None:
    if not path.exists():
        st.warning(f"Top feedback file が見つかりません: {path}")
        return

    df = pd.read_csv(path)
    if df.empty:
        st.info("大きな差分イベントは検出されませんでした。")
        return

    for rank, (_, row) in enumerate(df.head(max_items).iterrows(), start=1):
        with st.container(border=True):
            st.markdown(
                f"### {rank}. {row['time_range']}｜{row['body_part']}｜{row['metric']}"
            )
            st.metric("Difference score", f"{row['importance_score']:.1f}")
            st.write(row["comment"])


# Pose Snapshot controls helper
def show_pose_snapshot_controls(outputs: dict[str, Path], angle_threshold: float) -> None:
    video_path = outputs["input_video"]
    snapshot_output_dir = outputs["run_dir"] / "multi" / "snapshot"

    st.subheader("Pose Snapshot")
    st.write("指定した時刻で動画を止め、その瞬間の target / learner の骨格と角度差を比較します。")

    time_sec = st.number_input(
        "比較したい時刻 [sec]",
        min_value=0.0,
        value=0.0,
        step=0.1,
        format="%.1f",
    )

    snapshot_angle_threshold = st.slider(
        "Snapshot角度差ハイライト閾値 [deg]",
        min_value=5.0,
        max_value=60.0,
        value=float(angle_threshold),
        step=5.0,
    )

    if st.button("Pose Snapshotを生成", type="secondary"):
        try:
            snapshot_outputs = run_pose_snapshot(
                video_path=video_path,
                target_csv_path=outputs["person_0_pose_csv"],
                learner_csv_path=outputs["person_1_pose_csv"],
                output_dir=snapshot_output_dir,
                time_sec=time_sec,
                angle_diff_threshold_deg=snapshot_angle_threshold,
                decimals=3,
            )
            st.session_state["snapshot_outputs"] = snapshot_outputs
            st.success("Pose Snapshotを生成しました。")
        except Exception as error:
            st.exception(error)
            return

    snapshot_outputs = st.session_state.get("snapshot_outputs")
    if not snapshot_outputs:
        st.info("時刻を指定してPose Snapshotを生成してください。")
        return

    snapshot_image = snapshot_outputs["pose_snapshot_image"]
    snapshot_csv = snapshot_outputs["pose_snapshot_angles_csv"]

    if snapshot_image.exists():
        st.image(str(snapshot_image), caption="Pose Snapshot", use_container_width=True)

    if snapshot_csv.exists():
        st.subheader("角度差の詳細")
        snapshot_df = pd.read_csv(snapshot_csv)
        st.dataframe(snapshot_df, use_container_width=True)


def run_full_pipeline(
    input_video_path: Path,
    run_dir: Path,
    window_sec: float,
    angle_threshold: float,
    trail_sec: float,
    dynamics_threshold: float,
    trajectory_threshold_px: float,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}

    pose_output_dir = run_dir / "multi"
    feature_output_dirs = {
        0: pose_output_dir / "features_0",
        1: pose_output_dir / "features_1",
    }
    comparison_output_dir = pose_output_dir / "comparison"
    angle_output_dir = pose_output_dir / "angle_comparison"
    timeline_output_dir = pose_output_dir / "timeline"
    overlay_output_dir = pose_output_dir / "overlay"

    with st.status("1/7 人物検出と骨格推定を実行中...", expanded=True) as status:
        pose_outputs = run_multi_person_pose_extraction(
            video_path=input_video_path,
            output_dir=pose_output_dir,
            pose_model_path=POSE_MODEL_PATH,
            yolo_model_path=YOLO_MODEL_PATH,
            num_persons=2,
        )
        outputs.update(pose_outputs)
        status.update(label="1/7 人物検出と骨格推定が完了しました", state="complete")

    input_files = {
        0: pose_outputs["person_0_pose_csv"],
        1: pose_outputs["person_1_pose_csv"],
    }

    with st.status("2/7 特徴量を抽出中...", expanded=True) as status:
        feature_outputs = run_multi_feature_extraction(
            input_files=input_files,
            output_dirs=feature_output_dirs,
            window_sec=window_sec,
            create_plots=True,
            decimals=3,
        )
        outputs.update(feature_outputs)
        status.update(label="2/7 特徴量抽出が完了しました", state="complete")

    with st.status("3/7 軌道・緩急・部位別差分を比較中...", expanded=True) as status:
        comparison_outputs = run_dancer_comparison(
            person_0_csv=feature_outputs["person_0_window_feature_summary_csv"],
            person_1_csv=feature_outputs["person_1_window_feature_summary_csv"],
            output_dir=comparison_output_dir,
            decimals=3,
            create_plot=True,
        )
        outputs.update(comparison_outputs)
        status.update(label="3/7 特徴量比較が完了しました", state="complete")

    with st.status("4/7 角度差を比較中...", expanded=True) as status:
        angle_outputs = run_angle_comparison(
            target_csv_path=pose_outputs["person_0_pose_csv"],
            learner_csv_path=pose_outputs["person_1_pose_csv"],
            output_dir=angle_output_dir,
            window_sec=window_sec,
            angle_diff_threshold_deg=angle_threshold,
            decimals=3,
            create_plot=True,
        )
        outputs.update(angle_outputs)
        status.update(label="4/7 角度差比較が完了しました", state="complete")

    with st.status("5/7 Difference Timeline を生成中...", expanded=True) as status:
        timeline_outputs = run_difference_timeline(
            angle_summary_csv=angle_outputs["angle_difference_summary_csv"],
            body_part_difference_csv=comparison_outputs["body_part_difference_csv"],
            output_dir=timeline_output_dir,
            top_n=10,
            decimals=3,
            create_plot=True,
        )
        outputs.update(timeline_outputs)
        status.update(label="5/7 Difference Timeline が完成しました", state="complete")

    with st.status("6/7 角度差ハイライト動画を生成中...", expanded=True) as status:
        angle_overlay_outputs = create_angle_overlay_video(
            video_path=input_video_path,
            target_csv_path=pose_outputs["person_0_pose_csv"],
            learner_csv_path=pose_outputs["person_1_pose_csv"],
            output_dir=overlay_output_dir,
            angle_diff_threshold_deg=angle_threshold,
        )
        outputs.update(angle_overlay_outputs)
        status.update(label="6/7 角度差ハイライト動画が完成しました", state="complete")

    with st.status("7/7 軌道・緩急ハイライト動画を生成中...", expanded=True) as status:
        trajectory_overlay_outputs = create_trajectory_dynamics_overlay_video(
            video_path=input_video_path,
            target_csv_path=pose_outputs["person_0_pose_csv"],
            learner_csv_path=pose_outputs["person_1_pose_csv"],
            output_dir=overlay_output_dir,
            trail_sec=trail_sec,
            dynamics_diff_threshold_ratio=dynamics_threshold,
            trajectory_diff_threshold_px=trajectory_threshold_px,
        )
        outputs.update(trajectory_overlay_outputs)
        status.update(label="7/7 軌道・緩急ハイライト動画が完成しました", state="complete")

    outputs["run_dir"] = run_dir
    outputs["input_video"] = input_video_path
    return outputs


def main() -> None:
    st.title("Dance Motion Comparison")
    st.caption("目標ダンサーと学習者の動きの違いを、角度・軌道・緩急から可視化します。")

    with st.sidebar:
        st.header("設定")
        st.write("person_0 を target、person_1 を learner として扱います。")

        window_sec = st.slider(
            "解析ウィンドウ [sec]",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.5,
        )
        angle_threshold = st.slider(
            "角度差ハイライト閾値 [deg]",
            min_value=5.0,
            max_value=60.0,
            value=25.0,
            step=5.0,
        )
        trail_sec = st.slider(
            "軌跡表示時間 [sec]",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.5,
        )
        dynamics_threshold = st.slider(
            "速度差ハイライト閾値",
            min_value=0.1,
            max_value=1.0,
            value=0.35,
            step=0.05,
        )
        trajectory_threshold_px = st.slider(
            "軌道差ハイライト閾値 [px]",
            min_value=20.0,
            max_value=200.0,
            value=80.0,
            step=10.0,
        )

    if not POSE_MODEL_PATH.exists():
        st.error(f"MediaPipe model file が見つかりません: {POSE_MODEL_PATH}")
        st.stop()

    uploaded_video = st.file_uploader(
        "2人が映っているダンス動画をアップロードしてください",
        type=["mp4", "mov", "m4v", "avi"],
    )

    if uploaded_video is None:
        st.info("動画をアップロードすると解析を開始できます。")
        st.stop()

    st.subheader("入力動画")
    st.video(uploaded_video)

    if st.button("解析を実行", type="primary"):
        run_dir = create_run_dir()
        input_video_path = save_uploaded_video(uploaded_video, run_dir)

        st.session_state["last_run_dir"] = run_dir

        try:
            outputs = run_full_pipeline(
                input_video_path=input_video_path,
                run_dir=run_dir,
                window_sec=window_sec,
                angle_threshold=angle_threshold,
                trail_sec=trail_sec,
                dynamics_threshold=dynamics_threshold,
                trajectory_threshold_px=trajectory_threshold_px,
            )
            st.session_state["outputs"] = outputs
            st.success("解析が完了しました。")
        except Exception as error:
            st.exception(error)
            st.stop()

    outputs = st.session_state.get("outputs")
    if not outputs:
        st.stop()

    st.divider()
    st.header("解析結果")

    tab_summary, tab_snapshot, tab_video, tab_report, tab_csv = st.tabs(
        ["改善ポイント", "Pose Snapshot", "動画", "レポート", "CSV"]
    )

    with tab_summary:
        st.subheader("今回優先して直すポイント")
        show_top_feedback_cards(outputs["top_feedback_points_csv"], max_items=5)

        if outputs.get("difference_timeline_plot") and outputs["difference_timeline_plot"].exists():
            st.subheader("Difference Timeline")
            st.image(str(outputs["difference_timeline_plot"]), caption="差分が大きい時間帯")

        st.subheader("Difference Timeline Report")
        st.text(read_text_file(outputs["difference_timeline_report_txt"]))

    with tab_snapshot:
        show_pose_snapshot_controls(outputs, angle_threshold=angle_threshold)

    with tab_video:
        col1, col2 = st.columns(2)
        with col1:
            show_video(outputs["angle_overlay_video"], "角度差ハイライト動画")
        with col2:
            show_video(outputs["trajectory_dynamics_overlay_video"], "軌道・緩急ハイライト動画")

        st.subheader("YOLO + Pose 推定動画")
        show_video(outputs["annotated_video"], "人物検出・骨格推定結果")

    with tab_report:
        st.subheader("特徴量比較レポート")
        st.text(read_text_file(outputs["comparison_report_txt"]))

        st.subheader("角度差比較レポート")
        st.text(read_text_file(outputs["angle_difference_report_txt"]))

        col1, col2 = st.columns(2)
        with col1:
            if outputs.get("comparison_plot") and outputs["comparison_plot"].exists():
                st.image(str(outputs["comparison_plot"]), caption="特徴量差分")
        with col2:
            if outputs.get("angle_difference_plot") and outputs["angle_difference_plot"].exists():
                st.image(str(outputs["angle_difference_plot"]), caption="角度差分")

    with tab_csv:
        show_csv_preview(outputs["comparison_summary_csv"], "comparison_summary.csv")
        show_csv_preview(outputs["body_part_difference_csv"], "body_part_difference_summary.csv")
        show_csv_preview(outputs["angle_difference_events_csv"], "angle_difference_events.csv")
        show_csv_preview(outputs["difference_timeline_csv"], "difference_timeline.csv")
        show_csv_preview(outputs["top_feedback_points_csv"], "top_feedback_points.csv")

        st.subheader("出力フォルダ")
        st.code(str(outputs["run_dir"]))


if __name__ == "__main__":
    main()