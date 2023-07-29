"""Primary class for converting MoSeq Extraction data."""
from pynwb import NWBFile
from datetime import datetime
from pytz import timezone
import h5py
import numpy as np
import pandas as pd
from hdmf.backends.hdf5.h5_utils import H5DataIO
from neuroconv.basedatainterface import BaseDataInterface
from pynwb.core import DynamicTable, VectorData, VectorIndex
from pynwb.image import GrayscaleImage, ImageSeries, ImageMaskSeries
from pynwb import TimeSeries
from pynwb.base import Images
from pynwb.behavior import (
    CompassDirection,
    Position,
    SpatialSeries,
    BehavioralTimeSeries,
)
from neuroconv.tools import nwb_helpers
from ndx_events import LabeledEvents
from ndx_moseq import DepthImageSeries, MoSeqExtractGroup


class MoseqInterface(BaseDataInterface):
    """Moseq interface for markowitz_gillis_nature_2023 conversion"""

    def __init__(self, file_path: str, syllable_file_path: str):
        super().__init__(
            file_path=file_path,
            syllable_file_path=syllable_file_path,
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
            # Version
            version = np.array(file["metadata"]["extraction"]["extract_version"]).item().decode("ASCII")

            # Video
            processed_depth_video = np.array(file["frames"])
            loglikelihood_video = np.array(file["frames_mask"])

            # Timestamps
            TIMESTAMPS_TO_SECONDS = 1.25e-4
            timestamps = np.array(file["timestamps"]) * TIMESTAMPS_TO_SECONDS

            # Extraction
            background = np.array(file["metadata"]["extraction"]["background"])
            is_flipped = np.array(file["metadata"]["extraction"]["flips"])
            roi = np.array(file["metadata"]["extraction"]["roi"]) * 255
            true_depth = np.array(file["metadata"]["extraction"]["true_depth"]).item()

            # Kinematics
            kinematic_vars = {}
            for k, v in file["scalars"].items():
                kinematic_vars[k] = np.array(v)

            # Parameters
            parameter_names, parameter_data, parameter_descriptions = [], [], []
            for name, data in file["metadata"]["extraction"]["parameters"].items():
                if name == "output_dir":
                    continue  # skipping this bc it is Null
                parameter_names.append(name)
                if name == "input_file":
                    parameter_descriptions.append("Path to input depth video file")
                else:
                    parameter_descriptions.append(data.attrs["description"])
                data = np.array(data)
                if len(data.shape) == 0:
                    data = np.array([data.item()])
                parameter_data.append(data)

        kinect = nwbfile.create_device(name="kinect", manufacturer="Microsoft", description="Microsoft Kinect 2")
        processed_depth_video = (
            DepthImageSeries(  # TODO: add length and width px2mm conversions (length_mm / length_px, etc.)
                name="processed_depth_video",
                data=H5DataIO(processed_depth_video, compression=True),
                unit="millimeters",
                format="raw",
                timestamps=H5DataIO(timestamps, compression=True),  # TODO: All timestamps as links
                description="3D array of depth frames (nframes x w x h, in mm)",
                distant_depth=true_depth,
                device=kinect,
            )
        )
        loglikelihood_video = ImageMaskSeries(
            name="loglikelihood_video",
            data=H5DataIO(loglikelihood_video, compression=True),
            masked_imageseries=processed_depth_video,
            unit="a.u.",
            format="raw",
            timestamps=H5DataIO(timestamps, compression=True),
            description="Log-likelihood values from the tracking model (nframes x w x h)",
            device=kinect,
        )
        background = GrayscaleImage(
            name="background",
            data=H5DataIO(background, compression=True),
            description="Computed background image.",
        )
        roi = GrayscaleImage(  # TODO: ImageMask
            name="roi",
            data=H5DataIO(roi, compression=True),
            description="Computed region of interest.",
        )
        flipped_series = TimeSeries(
            name="flipped_series",
            data=H5DataIO(is_flipped, compression=True),
            unit="a.u.",
            timestamps=H5DataIO(timestamps, compression=True),
            description="Boolean array indicating whether the image was flipped left/right",
        )

        # Add Position Data
        position_data = np.vstack(
            (kinematic_vars["centroid_x_mm"], kinematic_vars["centroid_y_mm"], kinematic_vars["height_ave_mm"])
        ).T
        position_series = SpatialSeries(
            name="position",
            description="Position (x, y, height) in an open field.",
            data=H5DataIO(position_data, compression=True),
            timestamps=H5DataIO(timestamps, compression=True),
            reference_frame=metadata["Behavior"]["Position"]["reference_frame"],
            unit="mm",
        )
        position = Position(spatial_series=position_series, name="position")

        # Add Compass Direction Data
        heading_2d_series = SpatialSeries(
            name="heading_2d",
            description=(
                "The location of the mouse was identified by finding the centroid of the contour with the largest area "
                "using the OpenCV findcontours function. An 80×80 pixel bounding box was drawn around the "
                "identified centroid, and the orientation was estimated using an ellipse fit."
            ),
            data=H5DataIO(kinematic_vars["angle"], compression=True),
            timestamps=position_series.timestamps,
            reference_frame=metadata["Behavior"]["CompassDirection"]["reference_frame"],
            unit="radians",
        )
        heading_2d = CompassDirection(spatial_series=heading_2d_series, name="heading_2d")

        # Add speed/velocity data
        speed_2d_series = TimeSeries(
            name="speed_2d",
            description="2D speed (mm / frame), note that missing frames are not accounted for",
            data=H5DataIO(kinematic_vars["velocity_2d_mm"], compression=True),
            timestamps=H5DataIO(timestamps, compression=True),
            unit="mm/frame",
        )
        speed_2d = BehavioralTimeSeries(time_series=speed_2d_series, name="speed_2d")
        speed_3d_series = TimeSeries(
            name="speed_3d",
            description="3D speed (mm / frame), note that missing frames are not accounted for",
            data=H5DataIO(kinematic_vars["velocity_3d_mm"], compression=True),
            timestamps=H5DataIO(timestamps, compression=True),
            unit="mm/frame",
        )
        speed_3d = BehavioralTimeSeries(time_series=speed_3d_series, name="speed_3d")
        angular_velocity_2d_series = TimeSeries(
            name="angular_velocity_2d",
            description="Angular component of velocity (arctan(vel_x, vel_y))",
            data=H5DataIO(kinematic_vars["velocity_theta"], compression=True),
            timestamps=H5DataIO(timestamps, compression=True),
            unit="radians/frame",
        )
        angular_velocity_2d = BehavioralTimeSeries(time_series=angular_velocity_2d_series, name="angular_velocity_2d")

        # Add length/width/area data
        length_series = TimeSeries(
            name="length",
            description="Length of mouse (mm)",
            data=H5DataIO(kinematic_vars["length_mm"], compression=True),
            timestamps=H5DataIO(timestamps, compression=True),
            unit="mm",
        )
        length = BehavioralTimeSeries(time_series=length_series, name="length")
        width_series = TimeSeries(
            name="width",
            description="Width of mouse (mm)",
            data=H5DataIO(kinematic_vars["width_mm"], compression=True),
            timestamps=H5DataIO(timestamps, compression=True),
            unit="mm",
        )
        width = BehavioralTimeSeries(time_series=width_series, name="width")
        width_px_to_mm = kinematic_vars["width_mm"] / kinematic_vars["width_px"]
        length_px_to_mm = kinematic_vars["length_mm"] / kinematic_vars["length_px"]
        area_px_to_mm2 = width_px_to_mm * length_px_to_mm
        area_mm2 = kinematic_vars["area_px"] * area_px_to_mm2
        area_series = TimeSeries(
            name="area",
            description="Pixel-wise area of mouse (mm^2)",
            data=H5DataIO(area_mm2, compression=True),
            timestamps=H5DataIO(timestamps, compression=True),
            unit="mm^2",
        )
        area = BehavioralTimeSeries(time_series=area_series, name="area")

        # Combine all data into a behavioral processing module
        behavior_module = nwb_helpers.get_module(
            nwbfile,
            name="behavior",
            description="Processed behavioral data from MoSeq",
        )

        # Add Behavioral Syllables
        syllable_df = pd.read_csv(
            self.source_data["syllable_file_path"], sep=" ", header=None, names=["timestamp", "syllable"]
        )
        sorted_pseudoindex2name = metadata["BehavioralSyllable"]["sorted_pseudoindex2name"]
        id2sorted_index = metadata["BehavioralSyllable"]["id2sorted_index"]
        syllable_names = np.fromiter(sorted_pseudoindex2name.values(), dtype="O")
        syllable_pseudoindices = np.fromiter(sorted_pseudoindex2name.keys(), dtype=np.int64)
        index2name = syllable_names[np.argsort(syllable_pseudoindices)].tolist()
        for _ in range(len(id2sorted_index) - len(index2name)):
            index2name.append("Uncommon Syllable (frequency < 1%)")
        syllable_ids = syllable_df["syllable"]
        syllable_indices = syllable_ids.map(id2sorted_index).to_numpy(dtype=np.uint8)
        events = LabeledEvents(
            name="BehavioralSyllable",
            description="Behavioral Syllable identified by Motion Sequencing (MoSeq).",
            timestamps=H5DataIO(syllable_df["timestamp"].to_numpy(), compression=True),
            data=H5DataIO(syllable_indices, compression=True),
            labels=H5DataIO(index2name, compression=True),
        )
        nwbfile.add_acquisition(events)

        # Add Parameters
        parameter_set = DynamicTable(
            name="MoseqExtractParameterSet",
            description="Parameters used by moseq-extract.",
            id=[0],
        )
        for name, description, data in zip(parameter_names, parameter_descriptions, parameter_data):
            parameter_set.add_column(
                name=name,
                description=description,
                data=data,
                index=[data.shape[0]],
            )
        behavior_module.add(parameter_set)

        # Add MoseqExtractGroup
        moseq_extract_group = MoSeqExtractGroup(
            name="moseq_extract_group",
            version=version,
            background=background,
            processed_depth_video=processed_depth_video,
            loglikelihood_video=loglikelihood_video,
            roi=roi,
            flipped_series=flipped_series,
            depth_camera=kinect,
            position=position,
            heading_2d=heading_2d,
            speed_2d=speed_2d,
            speed_3d=speed_3d,
            angular_velocity_2d=angular_velocity_2d,
            length=length,
            width=width,
            area=area,
        )
        behavior_module.add(moseq_extract_group)
