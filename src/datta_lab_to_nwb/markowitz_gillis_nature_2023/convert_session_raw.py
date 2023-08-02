"""Primary script to run to convert an entire session for of data using the NWBConverter."""
from pathlib import Path
from typing import Union, Optional
from neuroconv.utils import dict_deep_update, load_dict_from_file
from datta_lab_to_nwb import markowitz_gillis_nature_2023
import shutil
from pynwb import NWBHDF5IO


def session_to_nwb(
    depth_path: Union[str, Path],
    metadata_path: Union[str, Path],
    timestamp_path: Union[str, Path],
    session_id: str,
    output_dir_path: Union[str, Path],
    ir_path: Optional[Union[str, Path]] = None,
    stub_test: bool = False,
):
    depth_path = Path(depth_path)
    metadata_path = Path(metadata_path)
    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    nwbfile_path = output_dir_path / f"{session_id}.nwb"
    source_data = {
        "DepthVideo": dict(
            data_path=str(depth_path),
            metadata_path=str(metadata_path),
            timestamp_path=timestamp_path,
        ),
    }
    conversion_options = {
        "DepthVideo": dict(),
    }
    if ir_path is not None:
        source_data["IRVideo"] = dict(
            data_path=str(ir_path),
            metadata_path=str(metadata_path),
            timestamp_path=timestamp_path,
        )
        conversion_options["IRVideo"] = dict()
    converter = markowitz_gillis_nature_2023.NWBConverter(source_data=source_data)
    metadata = converter.get_metadata()

    # Update metadata
    paper_metadata_path = Path(__file__).parent / "markowitz_gillis_nature_2023_metadata.yaml"
    paper_metadata = load_dict_from_file(paper_metadata_path)
    metadata = dict_deep_update(metadata, paper_metadata)

    # Run conversion
    converter.run_conversion(metadata=metadata, nwbfile_path=nwbfile_path, conversion_options=conversion_options)


if __name__ == "__main__":
    depth_path = "/Volumes/T7/CatalystNeuro/NWB/Datta/xtra_raw/session_20210215162554-455929/depth.avi"
    metadata_path = "/Volumes/T7/CatalystNeuro/NWB/Datta/xtra_raw/session_20210215162554-455929/proc/results_00.h5"
    timestamp_path = "/Volumes/T7/CatalystNeuro/NWB/Datta/xtra_raw/session_20210215162554-455929/depth_ts.txt"
    ir_path = "/Volumes/T7/CatalystNeuro/NWB/Datta/xtra_raw/session_20210215162554-455929/ir.avi"
    session_id = "874e5509-f12b-4aab-9a0e-64d004007a4f"
    output_dir_path = Path("/Volumes/T7/CatalystNeuro/NWB/Datta/conversion_nwb/")
    if output_dir_path.exists():
        shutil.rmtree(output_dir_path)
    stub_test = False
    session_to_nwb(
        depth_path=depth_path,
        metadata_path=metadata_path,
        timestamp_path=timestamp_path,
        ir_path=ir_path,
        session_id=session_id,
        output_dir_path=output_dir_path,
        stub_test=False,
    )
    nwbfile_path = Path(output_dir_path) / f"{session_id}.nwb"
    with NWBHDF5IO(nwbfile_path, "r") as io:
        nwbfile = io.read()
        print(nwbfile)
