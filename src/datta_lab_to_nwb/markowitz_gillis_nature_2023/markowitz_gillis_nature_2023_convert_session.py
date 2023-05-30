"""Primary script to run to convert an entire session for of data using the NWBConverter."""
from pathlib import Path
import shutil
from typing import Union
from datta_lab_to_nwb.markowitz_gillis_nature_2023.reproduce_figures import reproduce_fig1d
from datta_lab_to_nwb.markowitz_gillis_nature_2023 import MarkowitzGillisNature2023NWBConverter


def session_to_nwb(data_path: Union[str, Path], output_dir_path: Union[str, Path], stub_test: bool = False):
    data_path = Path(data_path)
    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    session_id = "2891f649-4fbd-4119-a807-b8ef507edfab"
    nwbfile_path = output_dir_path / f"{session_id}.nwb"

    source_data = dict()
    conversion_options = dict()

    # Add Behavior
    source_data.update(
        dict(
            Behavior=dict(
                file_path=str(data_path),
                session_uuid=session_id,
            )
        )
    )
    conversion_options.update(dict(Behavior=dict()))

    converter = MarkowitzGillisNature2023NWBConverter(source_data=source_data)
    metadata = converter.get_metadata()

    # # Update default metadata with the editable in the corresponding yaml file
    # editable_metadata_path = Path(__file__).parent / "markowitz_gillis_nature_2023_metadata.yaml"
    # editable_metadata = load_dict_from_file(editable_metadata_path)
    # metadata = dict_deep_update(metadata, editable_metadata)

    # Run conversion
    converter.run_conversion(metadata=metadata, nwbfile_path=nwbfile_path, conversion_options=conversion_options)


if __name__ == "__main__":
    # Parameters for conversion
    file_path = Path(
        "/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior/dlight_raw_data/dlight_photometry_processed_full.parquet"
    )
    output_dir_path = Path("/Volumes/T7/CatalystNeuro/NWB/Datta/conversion_nwb/")
    shutil.rmtree(output_dir_path)
    stub_test = False
    example_session = "2891f649-4fbd-4119-a807-b8ef507edfab"

    session_to_nwb(
        data_path=file_path,
        output_dir_path=output_dir_path,
        stub_test=stub_test,
    )
    nwbfile_path = output_dir_path / f"{example_session}.nwb"
    reproduce_fig1d(nwbfile_path)
