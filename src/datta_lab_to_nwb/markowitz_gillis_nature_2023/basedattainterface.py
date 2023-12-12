"""Primary class for handling metadata non-specific to any other DataInterfaces."""
from pathlib import Path

import pandas as pd
import numpy as np
from neuroconv.basetemporalalignmentinterface import BaseTemporalAlignmentInterface
from neuroconv.utils import load_dict_from_file


class BaseDattaInterface(BaseTemporalAlignmentInterface):
    """Base interface for markowitz_gillis_nature_2023 conversion w/ non-specific metadata"""

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        session_metadata = load_dict_from_file(self.source_data["session_metadata_path"])
        session_metadata = session_metadata[self.source_data["session_uuid"]]
        subject_metadata = load_dict_from_file(self.source_data["subject_metadata_path"])
        subject_metadata = subject_metadata[session_metadata["subject_id"]]

        metadata["NWBFile"]["session_description"] = session_metadata["session_description"]
        metadata["NWBFile"]["session_start_time"] = session_metadata["session_start_time"]
        metadata["NWBFile"]["identifier"] = self.source_data["session_uuid"]
        metadata["NWBFile"]["session_id"] = self.source_data["session_id"]

        metadata["Subject"] = {}
        metadata["Subject"]["subject_id"] = session_metadata["subject_id"]
        metadata["Subject"]["sex"] = subject_metadata["sex"]

        if self.source_data["alignment_path"] is not None and Path(self.source_data["alignment_path"]).exists():
            alignment_df = pd.read_parquet(self.source_data["alignment_path"])
            metadata["Alignment"]["slope"] = alignment_df["slope"].iloc[0]
            metadata["Alignment"]["bias"] = alignment_df["bias"].iloc[0]
        else:
            metadata["Alignment"]["slope"] = 1.0
            metadata["Alignment"]["bias"] = 0.0

        return metadata

    def get_metadata_schema(self) -> dict:
        metadata_schema = super().get_metadata_schema()
        if self.source_data["alignment_path"] is None:
            return metadata_schema
        metadata_schema["Alignment"] = {
            "type": "object",
            "description": "Metadata for temporal alignment with photometry data.",
            "required": True,
            "properties": {
                "slope": {
                    "description": "Slope of the linear regression mapping from behavioral video indices to demodulated photometry indices.",
                    "required": True,
                    "type": "float",
                },
                "bias": {
                    "description": "Bias of the linear regression mapping from behavioral video indices to demodulated photometry indices.",
                    "required": True,
                    "type": "float",
                },
                "start_time": {
                    "description": "Start time offset of raw fiber photometry data relative to behavioral video.",
                    "required": True,
                    "type": "float",
                },
            },
        }
        return metadata_schema

    def set_aligned_timestamps(self, aligned_timestamps: np.ndarray) -> None:
        self.aligned_timestamps = aligned_timestamps

    def get_timestamps(self) -> np.ndarray:
        return self.aligned_timestamps
