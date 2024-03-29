"""Base class for converting raw video data."""

from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBFile
from neuroconv.datainterfaces import VideoInterface

from .basedattainterface import BaseDattaInterface
from .utils import convert_timestamps_to_seconds


class BaseVideoInterface(BaseDattaInterface):
    """Base video interface for markowitz_gillis_nature_2023 conversion."""

    def __init__(
        self,
        data_path: str,
        timestamp_path: str,
        session_uuid: str,
        session_id: str,
        session_metadata_path: str,
        subject_metadata_path: str,
        alignment_path: str = None,
    ):
        super().__init__(
            data_path=data_path,
            timestamp_path=timestamp_path,
            session_uuid=session_uuid,
            session_id=session_id,
            session_metadata_path=session_metadata_path,
            subject_metadata_path=subject_metadata_path,
            alignment_path=alignment_path,
        )

    def get_original_timestamps(self) -> np.ndarray:
        return pd.read_csv(self.source_data["timestamp_path"], header=None).to_numpy().squeeze()

    def align_timestamps(self, metadata: dict) -> np.ndarray:
        timestamps = self.get_original_timestamps()
        timestamps = convert_timestamps_to_seconds(timestamps=timestamps, metadata=metadata)

        self.set_aligned_timestamps(aligned_timestamps=timestamps)
        if self.source_data["alignment_path"] is not None:
            aligned_starting_time = (
                metadata["Alignment"]["bias"] / metadata["Constants"]["DEMODULATED_PHOTOMETRY_SAMPLING_RATE"]
            )
            self.set_aligned_starting_time(aligned_starting_time=aligned_starting_time)
        return self.aligned_timestamps

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict) -> None:
        timestamps = self.align_timestamps(metadata=metadata)

        if not Path(self.source_data["data_path"]).exists():
            return

        video_interface = VideoInterface(file_paths=[self.source_data["data_path"]], verbose=True)
        video_interface.set_aligned_timestamps(aligned_timestamps=[timestamps])
        video_interface.add_to_nwbfile(
            nwbfile=nwbfile,
            metadata=metadata,
            stub_test=False,
            external_mode=True,
        )
