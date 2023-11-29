from pathlib import Path
from typing import Union
from neuroconv.utils import load_dict_from_file
from tqdm import tqdm


def dataset_to_nwb(
    processed_path: Union[str, Path],
    raw_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    skip_sessions: set,
):
    processed_path = Path(processed_path)
    raw_dir_path = Path(raw_dir_path)
    output_dir_path = Path(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for experimental_folder in tqdm(list(raw_dir_path.iterdir())):
        if experimental_folder.is_dir() and experimental_folder.name not in skip_experiments:
            for session_folder in experimental_folder.iterdir():
                if session_folder.is_dir() and session_folder.name not in skip_sessions:
                    results_file = session_folder / "proc" / "results_00.yaml"
                    results = load_dict_from_file(results_file)
                    assert "uuid" in results, f"UUID not found in {results_file}"


if __name__ == "__main__":
    processed_path = Path("NWB/DattaConv/processed_data")
    raw_dir_path = Path("NWB/DattaConv/raw_data")
    output_dir_path = Path("NWB/DattaConv/conversion_output")
    skip_experiments = {
        "keypoint",  # no proc folder for keypoints
    }
    skip_sessions = {
        "session_20210420113646-974717",  # _aggregate_results_arhmm_photometry_excitation_pulsed_01: missing everything except depth video
        "session_20210309134748-687283",  # _aggregate_results_arhmm_excitation_03: missing everything except depth video
        "session_20210224083612-947426",  # _aggregate_results_arhmm_excitation_03: missing proc folder
        "session_20210224094428-535503",  # _aggregate_results_arhmm_excitation_03: missing proc folder
        "session_20210309120607-939403",  # _aggregate_results_arhmm_excitation_03: proc folder empty
        "session_20201109130417-162983",  # _aggregate_results_arhmm_excitation_01: proc folder empty
        "session_20220308114215-760303",  # _aggregate_results_arhmm_scalar_03: missing proc folder
        "session_20211217102637-612299",  # _aggregate_results_arhmm_photometry_06: missing everything except ir video
        "session_20211202155132-245700",  # _aggregate_results_arhmm_photometry_06: missing everything except ir video
        "session_20210128093041-475933",  # _aggregate_results_arhmm_photometry_02: missing everything except ir video
        "session_20210215185110-281693",  # _aggregate_results_arhmm_photometry_02: missing everything except ir video
        "session_20210208173229-833584",  # _aggregate_results_arhmm_photometry_02: missing everything except ir video
        "session_20210201115439-569392",  # _aggregate_results_arhmm_photometry_02: missing everything except ir video
        "session_20200729112540-313279",  # _aggregate_results_arhmm_07: missing everything except depth video
        "session_20200810085750-497237",  # _aggregate_results_arhmm_07: missing everything except depth video
        "session_20200730090228-985303",  # _aggregate_results_arhmm_07: missing everything except depth video
        "session_20201207093653-476370",  # _aggregate_results_arhmm_excitation_02: missing everything except depth video
        "session_20210426143230-310843",  # _aggregate_results_arhmm_09: missing everything except depth video
        "session_20210429135801-758690",  # _aggregate_results_arhmm_09: missing everything except depth video
        "session_20191111130454-333065",  # _aggregate_results_arhmm_05: missing proc folder
        "session_20191111130847-263894",  # _aggregate_results_arhmm_05: missing proc folder
    }
    dataset_to_nwb(processed_path, raw_dir_path, output_dir_path, skip_sessions)
