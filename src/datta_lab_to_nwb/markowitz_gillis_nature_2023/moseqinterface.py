"""Primary class for converting MoSeq Extraction data."""
from typing import Literal, Optional
import numpy as np
from pynwb import NWBFile
from datetime import datetime
from pytz import timezone
import h5py
import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO
from neuroconv.basedatainterface import BaseDataInterface
from pynwb.image import GrayscaleImage, ImageSeries, ImageMaskSeries
from pynwb import TimeSeries
from pynwb.base import Images
from pynwb.behavior import (
    CompassDirection,
    Position,
    SpatialSeries,
)
from neuroconv.tools import nwb_helpers


class MoseqInterface(BaseDataInterface):
    """Moseq interface for markowitz_gillis_nature_2023 conversion"""

    def __init__(self, file_path: str):
        super().__init__(
            file_path=file_path,
        )

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        with h5py.File(self.source_data["file_path"]) as file:
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
        metadata["Subject"]["sex"] = "U"  # TODO: figure out how to get this

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
        with h5py.File(self.source_data["file_path"]) as file:
            # Video
            moseq_video = np.array(file["frames"])
            loglikelihood_video = np.array(file["frames_mask"])

            # Timestamps
            timestamps = np.array(file["timestamps"])

            # Extraction
            background = np.array(file["metadata"]["extraction"]["background"])
            is_flipped = np.array(file["metadata"]["extraction"]["flips"])
            roi = np.array(file["metadata"]["extraction"]["roi"]) * 255
            true_depth = np.array(file["metadata"]["extraction"]["true_depth"]).item()

            # Kinematics
            kinematic_vars = {}
            for k in file["scalars"].keys():
                kinematic_vars[k] = np.array(file["scalars"][k])

        kinect = nwbfile.create_device(name="kinect", manufacturer="Microsoft", description="Microsoft Kinect 2")
        moseq_video = ImageSeries(
            name="moseq_video",
            data=moseq_video,
            unit="millimeters",
            format="raw",
            timestamps=timestamps,
            description="3D array of depth frames (nframes x w x h, in mm)",
            comments=f"Detected true depth of arena floor in mm: {true_depth}",
            device=kinect,
        )
        loglikelihood_video = ImageMaskSeries(
            name="loglikelihood_video",
            data=loglikelihood_video,
            masked_imageseries=moseq_video,
            unit="a.u.",
            format="raw",
            timestamps=timestamps,
            description="Log-likelihood values from the tracking model (nframes x w x h)",
            device=kinect,
        )
        background = GrayscaleImage(
            name="background",
            data=background,
            description="Computed background image",
        )
        roi = GrayscaleImage(  # TODO: Ask about ImageMask
            name="roi",
            data=roi,
            description="Computed region of interest",
        )
        summary_images = Images(
            name="summary_images",
            images=[background, roi],
            description="Summary images from MoSeq",
        )
        flipped_series = TimeSeries(
            name="flipped_series",
            data=is_flipped,
            unit="a.u.",
            timestamps=timestamps,
            description="Boolean array indicating whether the image was flipped left/right",
        )
        nwbfile.add_acquisition(moseq_video)
        nwbfile.add_acquisition(loglikelihood_video)
        nwbfile.add_acquisition(summary_images)
        nwbfile.add_acquisition(flipped_series)

        # Add Position Data
        position_data = np.vstack(
            (kinematic_vars["centroid_x_mm"], kinematic_vars["centroid_y_mm"], kinematic_vars["height_ave_mm"])
        ).T
        position_spatial_series = SpatialSeries(
            name="SpatialSeries",
            description="Position (x, y, height) in an open field.",
            data=H5DataIO(position_data, compression=True),
            timestamps=H5DataIO(timestamps, compression=True),
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
            data=H5DataIO(kinematic_vars["angle"], compression=True),
            timestamps=position_spatial_series.timestamps,
            reference_frame=metadata["Behavior"]["CompassDirection"]["reference_frame"],
            unit="radians",
        )
        direction = CompassDirection(spatial_series=direction_spatial_series, name="CompassDirection")

        # Combine all data into a behavioral processing module
        behavior_module = nwb_helpers.get_module(nwbfile, name="behavior", description="Processed behavioral data")
        behavior_module.add(position)
        behavior_module.add(direction)
