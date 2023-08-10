"""Primary script to run to convert an entire session for of data using the NWBConverter."""
# Standard Library
from pathlib import Path
import shutil
from typing import Union, Literal

# Third Party
from neuroconv.utils import dict_deep_update, load_dict_from_file
from pynwb import NWBHDF5IO

# Local
from datta_lab_to_nwb.markowitz_gillis_nature_2023.postconversion import reproduce_fig1d
from datta_lab_to_nwb import markowitz_gillis_nature_2023


def session_to_nwb(
    session_id: str,
    processed_path: Union[str, Path],
    raw_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    experiment_type: Literal["reinforcement", "photometry", "reinforcement_photometry", "velocity_modulation"],
    stub_test: bool = False,
):
    processed_path = Path(processed_path)
    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    nwbfile_path = output_dir_path / f"{session_id}.nwb"
    photometry_path = processed_path / "dlight_raw_data/dlight_photometry_processed_full.parquet"
    if experiment_type == "velocity_modulation":
        optoda_path = processed_path / "optoda_raw_data/closed_loop_behavior_velocity_conditioned.parquet"
    else:
        optoda_path = processed_path / "optoda_raw_data/closed_loop_behavior.parquet"
    metadata_path = processed_path / "metadata"
    session_metadata_path = metadata_path / f"{experiment_type}_session_metadata.yaml"
    subject_metadata_path = metadata_path / f"{experiment_type}_subject_metadata.yaml"
    session_metadata = load_dict_from_file(session_metadata_path)
    session_metadata = session_metadata[session_id]
    raw_path = Path(raw_path)
    depth_path = raw_path / "depth.avi"
    depth_ts_path = raw_path / "depth_ts.txt"
    moseq_path = raw_path / "proc/results_00.h5"

    source_data, conversion_options = {}, {}
    if "reinforcement" in session_metadata.keys():
        source_data["Optogenetic"] = dict(
            file_path=str(optoda_path),
            session_metadata_path=str(session_metadata_path),
            subject_metadata_path=str(subject_metadata_path),
            session_uuid=session_id,
        )
        conversion_options["Optogenetic"] = {}
        behavioral_syllable_path = optoda_path
    if "photometry" in session_metadata.keys():
        tdt_path = list(raw_path.glob("tdt_data*.dat"))[0]
        tdt_metadata_path = list(raw_path.glob("tdt_data*.json"))[0]
        ir_path = raw_path / "ir.avi"
        source_data["FiberPhotometry"] = dict(
            file_path=str(photometry_path),
            tdt_path=str(tdt_path),
            tdt_metadata_path=str(tdt_metadata_path),
            session_metadata_path=str(session_metadata_path),
            subject_metadata_path=str(subject_metadata_path),
            session_uuid=session_id,
        )
        conversion_options["FiberPhotometry"] = {}
        behavioral_syllable_path = photometry_path  # Note: if photometry and optogenetics are both present, photometry is used for syllable data bc it is quicker to load
        source_data["IRVideo"] = dict(
            data_path=str(ir_path),
            timestamp_path=str(depth_ts_path),
            session_metadata_path=str(session_metadata_path),
            subject_metadata_path=str(subject_metadata_path),
            session_uuid=session_id,
        )
        conversion_options["IRVideo"] = {}
    source_data.update(
        dict(
            MoseqExtract=dict(
                file_path=str(moseq_path),
                session_metadata_path=str(session_metadata_path),
                subject_metadata_path=str(subject_metadata_path),
                session_uuid=session_id,
            ),
            BehavioralSyllable=dict(
                file_path=str(behavioral_syllable_path),
                session_metadata_path=str(session_metadata_path),
                subject_metadata_path=str(subject_metadata_path),
                session_uuid=session_id,
            ),
            DepthVideo=dict(
                data_path=str(depth_path),
                timestamp_path=str(depth_ts_path),
                session_metadata_path=str(session_metadata_path),
                subject_metadata_path=str(subject_metadata_path),
                session_uuid=session_id,
            ),
        )
    )
    conversion_options.update(
        dict(
            MoseqExtract={},
            BehavioralSyllable={},
            DepthVideo={},
        )
    )
    if experiment_type == "velocity_modulation":
        conversion_options["BehavioralSyllable"] = dict(velocity_modulation=True)

    converter = markowitz_gillis_nature_2023.NWBConverter(source_data=source_data)
    metadata = converter.get_metadata()

    # Update metadata
    paper_metadata_path = Path(__file__).parent / "markowitz_gillis_nature_2023_metadata.yaml"
    paper_metadata = load_dict_from_file(paper_metadata_path)
    metadata = dict_deep_update(metadata, paper_metadata)

    # Run conversion
    converter.run_conversion(metadata=metadata, nwbfile_path=nwbfile_path, conversion_options=conversion_options)


if __name__ == "__main__":
    # Parameters for conversion
    processed_path = Path("/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior")
    raw_rp_path = Path("/Volumes/T7/CatalystNeuro/NWB/Datta/xtra_raw/session_20210215162554-455929")
    raw_velocity_path = Path("/Volumes/T7/CatalystNeuro/NWB/Datta/xtra_raw/velocity_modulation")
    output_dir_path = Path("/Volumes/T7/CatalystNeuro/NWB/Datta/conversion_nwb/")
    if output_dir_path.exists():
        shutil.rmtree(
            output_dir_path, ignore_errors=True
        )  # ignore errors due to MacOS race condition (https://github.com/python/cpython/issues/81441)
    stub_test = False

    # Example UUIDs
    dls_dlight_1_example = "18dc5ad5-13f0-4297-8b21-75d434770e57"
    photometry_examples = [dls_dlight_1_example]
    reinforcement_example = "dcf0767a-b75d-4c79-a242-84dd5b5cdd00"
    excitation_example = "380d4711-85a6-4672-ad48-76e91607c41f"
    excitation_pulsed_example = "be01945e-c6d0-4bca-bd56-4d4466d9d832"
    reinforcement_examples = [reinforcement_example, excitation_example, excitation_pulsed_example]
    figure1d_example = "2891f649-4fbd-4119-a807-b8ef507edfab"
    pulsed_photometry_example = "b8360fcd-acfd-4414-9e67-ba0dc5c979a8"
    excitation_photometry_example = "95bec433-2242-4276-b8a5-6d069afa3910"
    reinforcement_photometry_examples = [figure1d_example, pulsed_photometry_example, excitation_photometry_example]
    raw_rp_example = "b814a426-7ec9-440e-baaa-105ba27a5fa6"
    velocity_modulation_example = "c621e134-50ec-4e8b-8175-a8c023d92789"

    experiment_type2example_sessions = {
        "reinforcement_photometry": [raw_rp_example],
        "velocity_modulation": [velocity_modulation_example],
    }
    experiment_type2raw_path = {
        "reinforcement_photometry": raw_rp_path,
        "velocity_modulation": raw_velocity_path,
    }
    for experiment_type, example_sessions in experiment_type2example_sessions.items():
        for example_session in example_sessions:
            session_to_nwb(
                session_id=example_session,
                processed_path=processed_path,
                raw_path=experiment_type2raw_path[experiment_type],
                output_dir_path=output_dir_path,
                experiment_type=experiment_type,
                stub_test=stub_test,
            )
    with NWBHDF5IO(output_dir_path / f"{raw_rp_example}.nwb", "r") as io:
        nwbfile = io.read()
        print(nwbfile)
    # nwbfile_path = output_dir_path / f"{figure1d_example}.nwb"
    # paper_metadata_path = Path(__file__).parent / "markowitz_gillis_nature_2023_metadata.yaml"
    # reproduce_figures.reproduce_fig1d(nwbfile_path, paper_metadata_path)
