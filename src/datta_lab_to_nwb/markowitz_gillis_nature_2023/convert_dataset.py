import traceback
import json
from pathlib import Path
from typing import Union
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from neuroconv.utils import load_dict_from_file
from tqdm import tqdm

from datta_lab_to_nwb.markowitz_gillis_nature_2023.convert_session import _safe_session_to_nwb

folder_name_to_experiment_type = {
    "_aggregate_results_arhmm_03": "reinforcement",
    "_aggregate_results_arhmm_04": "reinforcement",
    "_aggregate_results_arhmm_05": "reinforcement",
    "_aggregate_results_arhmm_06": "reinforcement",
    "_aggregate_results_arhmm_07": "reinforcement",
    "_aggregate_results_arhmm_08": "reinforcement",
    "_aggregate_results_arhmm_09": "reinforcement",
    "_aggregate_results_arhmm_11": "reinforcement",
    "_aggregate_results_arhmm_photometry_02": "reinforcement-photometry",
    "_aggregate_results_arhmm_photometry_03": "reinforcement-photometry",
    "_aggregate_results_arhmm_scalar_01": "velocity-modulation",
    "_aggregate_results_arhmm_scalar_03": "velocity-modulation",
    "_aggregate_results_arhmm_excitation_01": "reinforcement",
    "_aggregate_results_arhmm_excitation_02": "reinforcement",
    "_aggregate_results_arhmm_excitation_03": "reinforcement",
    "_aggregate_results_arhmm_photometry_excitation_02": "reinforcement-photometry",
    "_aggregate_results_arhmm_excitation_pulsed_01": "reinforcement",
    "_aggregate_results_arhmm_photometry_excitation_pulsed_01": "reinforcement",
    "_aggregate_results_arhmm_photometry_06": "photometry",
    "_aggregate_results_arhmm_photometry_07": "photometry",
    "_aggregate_results_arhmm_photometry_08": "photometry",
}


def get_all_processed_uuids(
    *,
    processed_path: Union[str, Path],
    output_dir_path: Union[str, Path],
) -> set:
    processed_path = Path(processed_path)
    output_dir_path = Path(output_dir_path)

    uuid_file_path = output_dir_path.parent / "all_processed_uuids.txt"

    if uuid_file_path.exists():
        with open(file=uuid_file_path, mode="r") as io:
            all_processed_uuids = set(json.load(fp=io))
        return all_processed_uuids

    photometry_uuids = pd.read_parquet(
        processed_path / "dlight_raw_data/dlight_photometry_processed_full.parquet", columns=["uuid"]
    )
    unique_photometry_uuids = set(photometry_uuids["uuid"])
    del photometry_uuids

    reinforcement_uuids = pd.read_parquet(
        processed_path / "optoda_raw_data/closed_loop_behavior.parquet", columns=["uuid"]
    )
    unique_reinforcement_uuids = set(reinforcement_uuids["uuid"])
    del reinforcement_uuids

    velocity_uuids = pd.read_parquet(
        processed_path / "optoda_raw_data/closed_loop_behavior_velocity_conditioned.parquet", columns=["uuid"]
    )
    unique_velocity_uuids = set(velocity_uuids["uuid"])
    del velocity_uuids

    all_processed_uuids = unique_photometry_uuids | unique_reinforcement_uuids | unique_velocity_uuids

    with open(file=uuid_file_path, mode="w") as io:
        json.dump(obj=list(all_processed_uuids), fp=io)
    return all_processed_uuids


def dataset_to_nwb(
    *,
    processed_path: Union[str, Path],
    raw_dir_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    skip_sessions: set,
    number_of_jobs: int,
    num_sessions_per_experiment: int = None,
):
    processed_path = Path(processed_path)
    raw_dir_path = Path(raw_dir_path)
    output_dir_path = Path(output_dir_path)

    log_folder_path = output_dir_path.parent / "logs"
    log_folder_path.mkdir(exist_ok=True)

    missing_folder_path = output_dir_path.parent / "missing"
    missing_folder_path.mkdir(exist_ok=True)

    all_processed_uuids = get_all_processed_uuids(processed_path=processed_path, output_dir_path=output_dir_path)

    experimental_folders = [
        folder
        for folder in raw_dir_path.iterdir()
        if folder.is_dir() and folder.name not in skip_experiments and folder.name.startswith("_")
    ]
    for experimental_folder in tqdm(iterable=experimental_folders, position=0, description="Converting experiments..."):
        experiment_type = folder_name_to_experiment_type[experimental_folder.name]
        session_folders = [
            folder for folder in experimental_folder.iterdir() if folder.is_dir() and folder.name not in skip_sessions
        ]
        if num_sessions_per_experiment is None:
            num_sessions_per_experiment = len(session_folders) + 1
        session_num = 0

        futures = list()
        with ProcessPoolExecutor(max_workers=number_of_jobs) as executor:
            for session_folder in session_folders:
                error_identifier_base = f"{experimental_folder.name}_{session_folder.name}"

                results_file = session_folder / "proc" / "results_00.yaml"

                if not results_file.exists():
                    (missing_folder_path / f"{error_identifier_base}.txt").touch()
                    continue

                results = load_dict_from_file(results_file)
                session_uuid = results["uuid"]
                if session_uuid not in all_processed_uuids:
                    continue

                futures.append(
                    executor.submit(
                        _safe_session_to_nwb,
                        session_uuid=session_uuid,
                        experiment_type=experiment_type,
                        processed_path=processed_path,
                        raw_path=session_folder,
                        output_dir_path=output_dir_path,
                        log_file_path=log_folder_path / f"{error_identifier_base}_{session_uuid}.txt",
                    ),
                )

                session_num += 1
                if session_num >= num_sessions_per_experiment:
                    break

            parallel_iterable = tqdm(
                iterable=as_completed(futures), position=1, description="Converting sessionsin parallel..."
            )
            for _ in parallel_iterable:
                pass


if __name__ == "__main__":
    number_of_jobs = 4

    processed_path = Path("E:/Datta/dopamine-reinforces-spontaneous-behavior")
    raw_dir_path = Path("E:/Datta")
    output_dir_path = Path("E:/datta_output/files")

    skip_experiments = {
        "keypoint",  # no proc folder for keypoints
    }
    temporary_skip_sessions = {
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
        "session_20200720110309-817092",
        "session_20210115130943-880998",
    }
    dataset_to_nwb(
        processed_path=processed_path,
        raw_dir_path=raw_dir_path,
        output_dir_path=output_dir_path,
        skip_sessions=temporary_skip_sessions,
        number_of_jobs=number_of_jobs,
        num_sessions_per_experiment=1,
    )
