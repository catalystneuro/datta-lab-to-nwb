"""Base class for converting raw video data."""
from pynwb import NWBFile
from datetime import datetime
from pytz import timezone
import h5py
import numpy as np
import pandas as pd
from neuroconv.datainterfaces import VideoInterface
from .basedattainterface import BaseDattaInterface


class BaseVideoInterface(BaseDattaInterface):
    """Base video interface for markowitz_gillis_nature_2023 conversion"""

    def __init__(
        self,
        data_path: str,
        timestamp_path: str,
        session_uuid: str,
        session_metadata_path: str,
        subject_metadata_path: str,
    ):
        super().__init__(
            data_path=data_path,
            timestamp_path=timestamp_path,
            session_uuid=session_uuid,
            session_metadata_path=session_metadata_path,
            subject_metadata_path=subject_metadata_path,
        )

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict) -> None:
        timestamps = pd.read_csv(self.source_data["timestamp_path"]).to_numpy().squeeze()
        TIMESTAMPS_TO_SECONDS = 1.25e-4
        timestamps -= timestamps[0]
        timestamps = timestamps * TIMESTAMPS_TO_SECONDS

        video_interface = VideoInterface(file_paths=[self.source_data["data_path"]], verbose=True)
        video_interface.set_aligned_timestamps(aligned_timestamps=[timestamps])
        video_interface.add_to_nwbfile(
            nwbfile=nwbfile,
            metadata=metadata,
            stub_test=True,
            external_mode=False,
        )
