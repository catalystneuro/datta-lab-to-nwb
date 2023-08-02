"""Primary class for converting raw depth video data."""
from pynwb import NWBFile
from datetime import datetime
from pytz import timezone
import h5py
import numpy as np
import pandas as pd
from hdmf.backends.hdf5.h5_utils import H5DataIO
from neuroconv.basedatainterface import BaseDataInterface
from pynwb.image import GrayscaleImage, ImageMaskSeries
from pynwb import TimeSeries
from pynwb.behavior import (
    CompassDirection,
    Position,
    SpatialSeries,
    BehavioralTimeSeries,
)
from neuroconv.tools import nwb_helpers
from neuroconv.datainterfaces import VideoInterface
from ndx_moseq import DepthImageSeries, MoSeqExtractGroup, MoSeqExtractParameterGroup
from typing import Optional


class DepthVideoInterface(BaseDataInterface):
    """Depth video interface for markowitz_gillis_nature_2023 conversion"""

    def __init__(self, depth_path: str, metadata_path: str, timestamp_path: str, ir_path: Optional[str] = None):
        super().__init__(
            depth_path=depth_path,
            metadata_path=metadata_path,
            timestamp_path=timestamp_path,
            ir_path=ir_path,
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

        file_paths = [self.source_data["depth_path"]]
        aligned_timestamps = [timestamps]
        starting_frames = [0]
        video_metadata = dict(
            Behavior=dict(
                Videos=[
                    dict(
                        name="depth_video",
                        description=(
                            "Depth videos of mouse behaviour were acquired at 30Hz using a Kinect 2 for Windows "
                            "(Microsoft) using a custom user interface written in Python (similar to ref. 60) on a Linux "
                            "computer. For all OFA experiments, except where noted, mice were placed in a circular open "
                            "field (US Plastics 14317) in the dark for 30min per experiment, for 2 experiments per day. "
                            "As described previously, the open field was sanded and painted black with spray paint "
                            "(Acryli-Quik Ultra Flat Black; 132496) to eliminate reflective artefacts in the depth video."
                        ),
                        unit="mm",
                    )
                ]
            )
        )
        print(self.source_data["ir_path"])
        if self.source_data["ir_path"] is not None:
            print("this runs")
            file_paths.append(self.source_data["ir_path"])
            aligned_timestamps.append(timestamps)
            starting_frames.append(0)
            video_metadata["Behavior"]["Videos"].append(
                dict(
                    name="ir_video",
                    description=(
                        "To align photometry and behavioural data, a custom IR led-based synchronization system was "
                        "implemented. Two sets of 3 IR (850nm) LEDs (Mouser part # 720-SFH4550) were attached to the walls "
                        "of the recording bucket and directed towards the Kinect depth sensor. The signal used to power "
                        "the LEDs was digitally copied to the TDT. An Arduino was used to generate a sequence of pulses "
                        "for each LED set. One LED set transitioned between on and off states every 2s while the other "
                        "transitioned into an on state randomly every 2–5s and remained in the on state for 1s. "
                        "The sequences of on and off states of each LED set were detected in the photometry data acquired "
                        "with the TDT and IR videos captured by the Kinect. "
                        "The timestamps of the sequences were aligned across each recording modality and photometry "
                        "recordings were down sampled to 30Hz to match the depth video sampling rate."
                    ),
                    unit="n.a.",
                )
            )
        video_interface = VideoInterface(file_paths=file_paths, verbose=True)
        video_interface.set_aligned_timestamps(aligned_timestamps=aligned_timestamps)
        metadata.update(video_metadata)
        video_interface.add_to_nwbfile(
            nwbfile=nwbfile, metadata=metadata, stub_test=True, external_mode=True, starting_frames=starting_frames
        )
