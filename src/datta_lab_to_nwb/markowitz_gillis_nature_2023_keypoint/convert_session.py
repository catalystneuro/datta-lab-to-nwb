"""Primary script to run to convert an entire session for of data using the NWBConverter."""
# Standard Library
from pathlib import Path
import shutil
from typing import Union, Literal

# Third Party
from neuroconv.utils import dict_deep_update, load_dict_from_file
from pynwb import NWBHDF5IO

# Local
from datta_lab_to_nwb import markowitz_gillis_nature_2023_keypoint


def session_to_nwb(
    subject_id: str,
    raw_path: Union[str, Path],
    metadata_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    stub_test: bool = False,
):
    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    session_id = f"keypoint-{subject_id}"
    nwbfile_path = output_dir_path / f"{session_id}.nwb"
    session_metadata_path = metadata_path / f"keypoint-session-metadata.yaml"
    subject_metadata_path = metadata_path / f"keypoint-subject-metadata.yaml"
    session_metadata = load_dict_from_file(session_metadata_path)
    session_metadata = session_metadata[subject_id]
    raw_path = Path(raw_path)

    source_data, conversion_options = {}, {}
    # Photometry
    tdt_path = list(raw_path.glob("tdt_data*.dat"))[0]
    tdt_metadata_path = list(raw_path.glob("tdt_data*.json"))[0]
    source_data["FiberPhotometry"] = dict(
        tdt_path=str(tdt_path),
        tdt_metadata_path=str(tdt_metadata_path),
        session_metadata_path=str(session_metadata_path),
        subject_metadata_path=str(subject_metadata_path),
        session_uuid=subject_id,
        session_id=session_id,
    )
    conversion_options["FiberPhotometry"] = {}

    converter = markowitz_gillis_nature_2023_keypoint.NWBConverter(source_data=source_data)
    metadata = converter.get_metadata()

    # Update metadata
    paper_metadata_path = Path(__file__).parent / "markowitz_gillis_nature_2023_keypoint_metadata.yaml"
    paper_metadata = load_dict_from_file(paper_metadata_path)
    metadata = dict_deep_update(metadata, paper_metadata)

    # Run conversion
    converter.run_conversion(metadata=metadata, nwbfile_path=nwbfile_path, conversion_options=conversion_options)


if __name__ == "__main__":
    # Parameters for conversion
    base_raw_path = Path("/Volumes/T7/CatalystNeuro/NWB/Datta/xtra_raw")
    output_dir_path = Path("/Volumes/T7/CatalystNeuro/NWB/Datta/conversion_nwb/")
    metadata_path = Path("/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior/metadata")
    if output_dir_path.exists():
        shutil.rmtree(
            output_dir_path, ignore_errors=True
        )  # ignore errors due to MacOS race condition (https://github.com/python/cpython/issues/81441)
    stub_test = False

    # Example subjects
    example_subjects = ["dls-dlight-9"]
    for subject_id in example_subjects:
        raw_path = base_raw_path / ("photometry-" + subject_id)
        session_to_nwb(
            subject_id=subject_id,
            raw_path=raw_path,
            metadata_path=metadata_path,
            output_dir_path=output_dir_path,
            stub_test=stub_test,
        )
    with NWBHDF5IO(output_dir_path / f"keypoint-dls-dlight-9.nwb", "r") as io:
        nwbfile = io.read()
        print(nwbfile)
