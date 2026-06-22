from pathlib import Path
from typing import Dict

from feature_extraction import run_feature_extraction


INPUT_FILES = {
    0: Path("outputs/multi/person_0_pose.csv"),
    1: Path("outputs/multi/person_1_pose.csv"),
}

OUTPUT_DIRS = {
    0: Path("outputs/multi/features_0"),
    1: Path("outputs/multi/features_1"),
}

WINDOW_SEC = 2.0


def run_multi_feature_extraction(
    input_files: dict[int, Path] = INPUT_FILES,
    output_dirs: dict[int, Path] = OUTPUT_DIRS,
    window_sec: float = WINDOW_SEC,
    create_plots: bool = True,
    decimals: int = 3,
) -> Dict[str, Path]:
    """Run feature extraction for all detected people.

    Designed to be called from a Web app such as Streamlit.
    Returns paths to generated outputs.
    """

    outputs: Dict[str, Path] = {}

    for person_id in input_files:
        input_csv = input_files[person_id]
        output_dir = output_dirs[person_id]

        if not input_csv.exists():
            print(f"Skip person_{person_id}: file not found -> {input_csv}")
            continue

        print(f"Processing person_{person_id}...")

        person_outputs = run_feature_extraction(
            input_csv_path=input_csv,
            output_dir=output_dir,
            window_sec=window_sec,
            create_plots=create_plots,
            decimals=decimals,
        )

        for key, value in person_outputs.items():
            outputs[f"person_{person_id}_{key}"] = value

    print("Finished multi-person feature extraction.")
    return outputs



def main() -> None:
    run_multi_feature_extraction(
        window_sec=WINDOW_SEC,
        create_plots=True,
        decimals=3,
    )


if __name__ == "__main__":
    main()