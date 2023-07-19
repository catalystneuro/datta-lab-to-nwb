"""Primary class for converting in-vitro photometry data."""
import numpy as np
from pynwb import NWBFile
from neuroconv.datainterfaces import TiffImagingInterface
from neuroconv.tools import nwb_helpers
from neuroconv.utils import load_dict_from_file
from datetime import datetime
import pytz
from pathlib import Path


class HEKInterface(TiffImagingInterface):
    """HEK interface for markowitz_gillis_nature_2023 conversion"""
    ExtractorName = "TiffImagingExtractor"

    def __init__(self, file_path: str, scale_path: str, verbose: bool = False):
        sampling_frequency = 0.5
        super().__init__(
            file_path=file_path,
            sampling_frequency=sampling_frequency,
            verbose=verbose,
        )
        date = Path(file_path).name.split('_')[2]
        date = datetime.strptime(date, '%Y%m%d')
        time = Path(file_path).name.split('_')[3]
        time = datetime.strptime(time, '%H%M%S')
        timezone = pytz.timezone("America/New_York")
        self.source_data['session_start_time'] = datetime.combine(date, time.time(), tzinfo=timezone)

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        metadata["NWBFile"]["session_start_time"] = self.source_data['session_start_time']
        metadata["Subject"] = {}
        metadata["Subject"]["subject_id"] = "???"
        metadata["Subject"]["sex"] = "U"
        return metadata

    def run_conversion(self, nwbfile: NWBFile, metadata: dict) -> NWBFile:
        """Run conversion of data from the source file into the nwbfile."""
        nwbfile = super().run_conversion(nwbfile=nwbfile, metadata=metadata)

        return nwbfile
