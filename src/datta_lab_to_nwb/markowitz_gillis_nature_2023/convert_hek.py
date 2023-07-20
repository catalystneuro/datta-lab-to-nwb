"""Primary script to run to convert an entire session for of data using the NWBConverter."""
from pathlib import Path
from typing import Union
from neuroconv.utils import dict_deep_update, load_dict_from_file
from datta_lab_to_nwb import markowitz_gillis_nature_2023
import shutil
from pynwb import NWBHDF5IO
from datta_lab_to_nwb.markowitz_gillis_nature_2023.postconversion import reproduce_figS1abcd


def session_to_nwb(
    file_path: Union[str, Path],
    scale_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    stub_test: bool = False,
):
    file_path = Path(file_path)
    scale_path = Path(scale_path)
    output_dir_path = Path(output_dir_path)
    if stub_test:
        output_dir_path = output_dir_path / "nwb_stub"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    experiment_name = file_path.name.split("_")[1]
    nwbfile_path = output_dir_path / f"HEK_{experiment_name}.nwb"
    source_data = {
        "HEK": dict(file_path=str(file_path), scale_path=str(scale_path)),
    }
    conversion_options = {
        "HEK": dict(),
    }
    converter = markowitz_gillis_nature_2023.NWBConverter(source_data=source_data)
    metadata = converter.get_metadata()

    # Update metadata
    paper_metadata_path = Path(__file__).parent / "markowitz_gillis_nature_2023_metadata.yaml"
    paper_metadata = load_dict_from_file(paper_metadata_path)
    metadata = dict_deep_update(metadata, paper_metadata)

    # Run conversion
    converter.run_conversion(metadata=metadata, nwbfile_path=nwbfile_path, conversion_options=conversion_options)


if __name__ == "__main__":
    scale_path = "/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior/hek_raw_data/UFM-L_0.01mm_60X_bin2_20220511_093613.tif"
    file_paths = [
        "/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior/hek_raw_data/210716_JM_HEK293T_dLight/dLight_exp3_20210716_110931_sVG.tif",
        "/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior/hek_raw_data/210716_JM_HEK293T_dLight/dLight_exp4_20210716_112801_sVG.tif",
        "/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior/hek_raw_data/210716_JM_HEK293T_dLight/dLight_exp5_20210716_114158_sVG.tif",
    ]
    output_dir_path = Path("/Volumes/T7/CatalystNeuro/NWB/Datta/conversion_nwb/")
    if output_dir_path.exists():
        shutil.rmtree(output_dir_path)
    stub_test = False
    # file_paths = file_paths[:1]
    for file_path in file_paths:
        session_to_nwb(
            file_path=file_path,
            scale_path=scale_path,
            output_dir_path=output_dir_path,
            stub_test=False,
        )
    nwb_files = [Path(output_dir_path) / f"HEK_{Path(file_path).name.split('_')[1]}.nwb" for file_path in file_paths]
    with NWBHDF5IO(nwb_files[0], "r") as io:
        nwbfile = io.read()
        print(nwbfile)

    reproduce_figS1abcd.reproduce_figS1abcd(file_paths, nwb_files)
