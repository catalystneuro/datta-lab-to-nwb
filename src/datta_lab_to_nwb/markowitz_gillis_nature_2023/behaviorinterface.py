"""Primary class for converting experiment-specific behavior."""
import numpy as np
import pandas as pd
from pynwb import NWBFile
from pynwb.behavior import (
    CompassDirection,
    Position,
    SpatialSeries,
)
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools import nwb_helpers
from neuroconv.utils import load_dict_from_file
from hdmf.backends.hdf5.h5_utils import H5DataIO


class BehaviorInterface(BaseDataInterface):
    """Behavior interface for markowitz_gillis_nature_2023 conversion"""

    def __init__(self, file_path: str, session_uuid: str, session_metadata_path: str):
        # This should load the data lazily and prepare variables you need
        columns = (
            "uuid",
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
            session_metadata_path=session_metadata_path,
        )

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
            name="HeadOrientation",
            description=(
                "The location of the mouse was identified by finding the centroid of the contour with the largest area "
                "using the OpenCV findcontours function. An 80×80 pixel bounding box was drawn around the "
                "identified centroid, and the orientation was estimated using an ellipse fit."
            ),
            data=H5DataIO(session_df.angle_unwrapped.to_numpy(), compression=True),
            timestamps=position_spatial_series.timestamps,
            reference_frame=metadata["Behavior"]["CompassDirection"]["reference_frame"],
            unit="radians",
        )
        direction = CompassDirection(spatial_series=direction_spatial_series, name="CompassDirection")

        # Combine all data into a behavioral processing module
        behavior_module = nwb_helpers.get_module(nwbfile, name="behavior", description="Processed behavioral data")
        behavior_module.add(position)
        behavior_module.add(direction)

        return nwbfile
