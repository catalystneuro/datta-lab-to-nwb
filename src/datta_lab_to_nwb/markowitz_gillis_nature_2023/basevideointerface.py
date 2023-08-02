"""Base class for converting raw video data."""
from pynwb import NWBFile
from datetime import datetime
from pytz import timezone
import h5py
import numpy as np
import pandas as pd
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.datainterfaces import VideoInterface


class BaseVideoInterface(BaseDataInterface):
    """Base video interface for markowitz_gillis_nature_2023 conversion"""

    def __init__(self, data_path: str, metadata_path: str, timestamp_path: str):
        super().__init__(
            data_path=data_path,
            metadata_path=metadata_path,
            timestamp_path=timestamp_path,
        )

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        with h5py.File(self.source_data["metadata_path"]) as file:
            session_id = np.array(file["metadata"]["uuid"], dtype="U").item()
            subject_id = np.array(file["metadata"]["acquisition"]["SubjectName"], dtype="U").item()
            session_start_time = np.array(file["metadata"]["acquisition"]["StartTime"], dtype="U").item()
            session_name = np.array(file["metadata"]["acquisition"]["SessionName"], dtype="U").item()
        metadata["NWBFile"]["session_id"] = session_id
        metadata["NWBFile"]["identifier"] = session_id
        metadata["NWBFile"]["session_start_time"] = datetime.fromisoformat(session_start_time).astimezone(
            timezone("US/Eastern")
        )
        metadata["NWBFile"]["session_description"] = session_name
        metadata["Subject"]["subject_id"] = subject_id
        metadata["Subject"]["sex"] = "U"  # TODO: Add dict of sexes from email

        return metadata

    def get_metadata_schema(self) -> dict:
        metadata_schema = super().get_metadata_schema()
        metadata_schema["properties"]["Behavior"] = {
            "type": "object",
            "properties": {
                "CompassDirection": {
                    "type": "object",
                    "properties": {
                        "reference_frame": {"type": "string"},
                    },
                },
                "Position": {
                    "type": "object",
                    "properties": {
                        "reference_frame": {"type": "string"},
                    },
                },
            },
        }
        metadata_schema["properties"]["BehavioralSyllable"] = {
            "type": "object",
            "properties": {
                "sorted_pseudoindex2name": {"type": "object"},
                "id2sorted_index": {"type": "object"},
                "sorted_index2id": {"type": "object"},
            },
        }
        return metadata_schema

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
