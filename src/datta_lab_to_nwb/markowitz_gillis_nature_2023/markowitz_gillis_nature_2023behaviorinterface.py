"""Primary class for converting experiment-specific behavior."""
import numpy as np
import pandas as pd
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


class MarkowitzGillisNature2023BehaviorInterface(BaseDataInterface):
    """Behavior interface for markowitz_gillis_nature_2023 conversion"""

    def __init__(self, file_path: FilePath):
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
        metadata_columns = (
            "date",
            "mouse_id",
        )
        super().__init__(file_path=file_path, columns=columns, metadata_columns=metadata_columns)

    def get_metadata(self, session_uuid: str) -> dict:
        # TODO: store metadata in .yaml file
        metadata = super().get_metadata()

        # get session metadata TODO: move session metadata to a separate file to avoid multiple reads
        session_df = pd.read_parquet(
            self.source_data["file_path"], columns=self.source_data["columns"], filters=[("uuid", "==", session_uuid)]
        )
        for col in self.source_data["metadata_columns"]:
            first_notnull = session_df.loc[session_df[col].notnull(), col].iloc[0]
            metadata['NWBFile'][col] = first_notnull
        session_name = set(session_df.session_name[session_df.session_name.notnull()]) | set(
            session_df.SessionName[session_df.SessionName.notnull()]
        )
        assert len(session_name) == 1, "Multiple session names found"
        metadata['NWBFile']["session_name"] = session_name.pop()
        print(f"metadata: {metadata}")

        return metadata

    def run_conversion(self, nwbfile: NWBFile, metadata: dict, session_uuid: str) -> NWBFile:
        """Run conversion of data from the source file into the nwbfile."""
        session_df = pd.read_parquet(
            self.source_data["file_path"], columns=self.source_data["columns"], filters=[("uuid", "==", session_uuid)]
        )

        # Add Position Data
        position_data = np.vstack((session_df.centroid_x_mm, session_df.centroid_y_mm, session_df.height_ave_mm)).T
        position_spatial_series = SpatialSeries(
            name="SpatialSeries",
            description="Position (x, y, height) in an open field.",
            data=position_data,
            timestamps=session_df.timestamp.to_numpy(),
            reference_frame=metadata["Behavior"]["Position"]["reference_frame"],
            unit="mm",
        )
        position = Position(spatial_series=position_spatial_series)

        # Add Compass Direction Data
        direction_spatial_series = SpatialSeries(
            name="OrientationEllipse",
            description="Mouse orientation in radians estimated using an ellipse fit.",
            data=session_df.angle_unwrapped.to_numpy(),
            timestamps=position_spatial_series.timestamps,
            reference_frame=metadata["Behavior"]["CompassDirection"]["reference_frame"],
            unit="radians",
        )
        direction = CompassDirection(spatial_series=direction_spatial_series, name="CompassDirection")

        # Add Syllable Data
        syllable_time_series = TimeSeries(
            name="BehavioralSyllable",
            data=session_df["predicted_syllable (offline)"].to_numpy(),
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
