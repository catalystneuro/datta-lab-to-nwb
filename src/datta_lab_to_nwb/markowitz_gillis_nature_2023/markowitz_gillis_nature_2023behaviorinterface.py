"""Primary class for converting experiment-specific behavior."""
import numpy as np
import pandas as pd
import yaml
from pydantic import FilePath
from pynwb import NWBFile, TimeSeries
from pynwb.behavior import (
    BehavioralTimeSeries,
    CompassDirection,
    Position,
    SpatialSeries,
)
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools import nwb_helpers
from hdmf.backends.hdf5.h5_utils import H5DataIO


class MarkowitzGillisNature2023BehaviorInterface(BaseDataInterface):
    """Behavior interface for markowitz_gillis_nature_2023 conversion"""

    def __init__(self, file_path: str, session_uuid: str, metadata_path: str):
        # This should load the data lazily and prepare variables you need
        columns = (
            "uuid",
            "predicted_syllable (offline)",
            "centroid_x_mm",
            "centroid_y_mm",
            "height_ave_mm",
            "angle_unwrapped",
            "timestamp",
        )
        super().__init__(
            file_path=file_path,
            session_uuid=session_uuid,
            columns=columns,
            metadata_path=metadata_path,
        )

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        with open(self.source_data["metadata_path"]) as f:
            session_metadata = yaml.safe_load(f)
            session_metadata = session_metadata[self.source_data["session_uuid"]]
        metadata["NWBFile"]["session_description"] = session_metadata["session_description"]
        metadata["NWBFile"]["session_start_time"] = session_metadata["session_start_time"]
        metadata["Subject"] = {}
        metadata["Subject"]["subject_id"] = session_metadata["subject_id"]
        metadata["NWBFile"]["identifier"] = self.source_data["session_uuid"]
        metadata["NWBFile"]["session_id"] = self.source_data["session_uuid"]

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
                "Syllable": {
                    "type": "object",
                    "properties": {
                        "syllable_id2name": {"type": "object"},
                    },
                },
            },
        }
        return metadata_schema

    def run_conversion(self, nwbfile: NWBFile, metadata: dict) -> NWBFile:
        """Run conversion of data from the source file into the nwbfile."""
        session_df = pd.read_parquet(
            self.source_data["file_path"],
            columns=self.source_data["columns"],
            filters=[("uuid", "==", self.source_data["session_uuid"])],
        )

        # Add Position Data
        position_data = np.vstack((session_df.centroid_x_mm, session_df.centroid_y_mm, session_df.height_ave_mm)).T
        position_spatial_series = SpatialSeries(
            name="SpatialSeries",
            description="Position (x, y, height) in an open field.",
            data=H5DataIO(position_data, compression=True),
            timestamps=H5DataIO(session_df.timestamp.to_numpy(), compression=True),
            reference_frame=metadata["Behavior"]["Position"]["reference_frame"],
            unit="mm",
        )
        position = Position(spatial_series=position_spatial_series)

        # Add Compass Direction Data
        direction_spatial_series = SpatialSeries(
            name="OrientationEllipse",
            description="Mouse orientation in radians estimated using an ellipse fit.",
            data=H5DataIO(session_df.angle_unwrapped.to_numpy(), compression=True),
            timestamps=position_spatial_series.timestamps,
            reference_frame=metadata["Behavior"]["CompassDirection"]["reference_frame"],
            unit="radians",
        )
        direction = CompassDirection(spatial_series=direction_spatial_series, name="CompassDirection")

        # Add Syllable Data
        syllable_time_series = TimeSeries(
            name="BehavioralSyllable",
            data=H5DataIO(session_df["predicted_syllable (offline)"].to_numpy(), compression=True),
            timestamps=position_spatial_series.timestamps,
            description="Behavioral Syllable identified by Motion Sequencing (MoSeq).",
            unit="n.a.",
        )
        behavioral_time_series = BehavioralTimeSeries(
            time_series=syllable_time_series,
            name="SyllableTimeSeries",
        )

        # Combine all data into a behavioral processing module
        behavior_module = nwb_helpers.get_module(nwbfile, name="behavior", description="Processed behavioral data")
        behavior_module.add(position)
        behavior_module.add(direction)
        behavior_module.add(behavioral_time_series)

        return nwbfile
