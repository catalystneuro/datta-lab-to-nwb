from pathlib import Path
from typing import Union
from neuroconv.utils import load_dict_from_file


def dataset_to_nwb(processed_path: Union[str, Path], raw_dir_path: Union[str, Path], output_dir_path: Union[str, Path]):
    processed_path = Path(processed_path)
    raw_dir_path = Path(raw_dir_path)
    output_dir_path = Path(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for experimental_folder in raw_dir_path.iterdir():
        assert experimental_folder.is_dir(), f"{experimental_folder} is not a directory"
        for session_folder in experimental_folder.iterdir():
            assert session_folder.is_dir(), f"{session_folder} is not a directory"
            results_file = session_folder / "proc" / "results00.yaml"
            results = load_dict_from_file(results_file)
            assert "uuid" in results, f"UUID not found in {results_file}"


if __name__ == "__main__":
    processed_path = "/path/to/processed"
    raw_dir_path = "/path/to/raw"
    output_dir_path = "/path/to/output"
    dataset_to_nwb(processed_path, raw_dir_path, output_dir_path)
