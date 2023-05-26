"""Primary class for converting experiment-specific behavior."""
import numpy as np
import pandas as pd
from pynwb import NWBFile, TimeSeries
from pynwb.behavior import (
    BehavioralTimeSeries,
    CompassDirection,
    Position,
    SpatialSeries,
)
from neuroconv.basedatainterface import BaseDataInterface


class MarkowitzGillisNature2023BehaviorInterface(BaseDataInterface):
    """Behavior interface for markowitz_gillis_nature_2023 conversion"""

    def __init__(self, filename: str):
        # This should load the data lazily and prepare variables you need
        columns = (
            "predicted_syllable (offline)",
            "uuid",
            "date",
            "session_name",
            "SessionName",
            "mouse_id",
            "centroid_x_mm",
            "centroid_y_mm",
            "height_ave_mm",
            "angle_unwrapped",
            "timestamp",
        )
        metadata_columns = ["uuid", "date", "mouse_id"]
        super().__init__(filename=filename, columns=columns, metadata_columns=metadata_columns)

    def get_metadata(self, session_uuid: str) -> dict:
        """Get metadata about the experiment as a whole."""
        # TODO: store metadata in .yaml file
        metadata = super().get_metadata()

        # get session metadata TODO: move session metadata to a separate file to avoid multiple reads
        session_df = pd.read_parquet(
            self.source_data["filename"], columns=self.source_data["columns"], filters=[("uuid", "==", session_uuid)]
        )
        for col in self.source_data["metadata_columns"]:
            first_notnull = session_df.loc[session_df[col].notnull(), col].iloc[0]
            metadata[col] = first_notnull
        session_name = set(session_df.session_name[session_df.session_name.notnull()]) | set(
            session_df.SessionName[session_df.SessionName.notnull()]
        )
        assert len(session_name) == 1, "Multiple session names found"
        metadata["session_name"] = session_name.pop()

        return metadata

    def run_conversion(self, nwbfile: NWBFile, metadata: dict, session_uuid: str) -> NWBFile:
        """Run conversion of data from the source file into the nwbfile."""
        session_df = pd.read_parquet(
            self.source_data["filename"], columns=self.source_data["columns"], filters=[("uuid", "==", session_uuid)]
        )

        # Add Position Data
        position_data = np.vstack((session_df.centroid_x_mm, session_df.centroid_y_mm, session_df.height_ave_mm)).T
        position_spatial_series = SpatialSeries(
            name="SpatialSeries",
            description="Position (x, y, height) in an open field.",
            data=position_data,
            timestamps=session_df.timestamp.to_numpy(),
            reference_frame=metadata["reference_frame"],
            unit="mm",
        )
        position = Position(spatial_series=position_spatial_series)

        # Add Compass Direction Data
        direction_spatial_series = SpatialSeries(
            name="SpatialSeries",
            description="View angle of the subject measured in radians.",
            data=session_df.angle_unwrapped.to_numpy(),
            timestamps=session_df.timestamp.to_numpy(),
            reference_frame=metadata["reference_frame"],
            unit="radians",
        )
        direction = CompassDirection(spatial_series=direction_spatial_series, name="CompassDirection")

        # Add Syllable Data
        syllable_time_series = TimeSeries(
            name="syllable",
            data=session_df["predicted_syllable (offline)"].to_numpy(),
            timestamps=session_df.timestamp.to_numpy(),
            description="Behavioral Syllable identified by Motion Sequencing (MoSeq).",
            unit="None",
        )
        behavioral_time_series = BehavioralTimeSeries(
            time_series=syllable_time_series,
            name="SyllableTimeSeries",
        )

        # Combine all data into a behavioral processing module
        behavior_module = nwbfile.create_processing_module(name="behavior", description="Processed behavioral data")
        behavior_module.add(position)
        behavior_module.add(direction)
        behavior_module.add(behavioral_time_series)

        return nwbfile
