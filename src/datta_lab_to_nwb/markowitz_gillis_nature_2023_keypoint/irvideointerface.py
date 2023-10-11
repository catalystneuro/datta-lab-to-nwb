"""Base class for converting raw video data."""
from pynwb import NWBFile
import numpy as np
from pathlib import Path
from neuroconv.datainterfaces import VideoInterface
from ..markowitz_gillis_nature_2023.basedattainterface import BaseDattaInterface


class IRVideoInterface(BaseDattaInterface):
    """IR video interface for markowitz_gillis_nature_2023 conversion of keypoint data."""

    def __init__(
        self,
        data_path: str,
        session_uuid: str,
        session_id: str,
        session_metadata_path: str,
        subject_metadata_path: str,
    ):
        super().__init__(
            data_path=Path(data_path),
            session_uuid=session_uuid,
            session_id=session_id,
            session_metadata_path=session_metadata_path,
            subject_metadata_path=subject_metadata_path,
        )

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict) -> None:
        SAMPLING_RATE = 30
        matched_timestamp_path = (
            self.source_data["data_path"] / f"photometry-{self.source_data['session_uuid']}.matched_timestamps.npy"
        )
        matched_timestamps = np.load(matched_timestamp_path)
        assert np.all(
            np.diff(matched_timestamps, axis=0) == 1
        ), "Matched timestamp indices are not monotonically increasing"
        timestamp_offsets = matched_timestamps[0, :]
        camera_names = ["bottom", "side1", "side2", "side3", "side4", "top"]
        camera_paths, camera_timestamps = [], []
        for i, camera in enumerate(camera_names):
            timestamp_path = (
                self.source_data["data_path"]
                / f"photometry-{self.source_data['session_uuid']}.{camera}.system_timestamps.npy"
            )
            timestamps = np.load(timestamp_path)
            timestamps = timestamps + (np.max(timestamp_offsets) - timestamp_offsets[i]) * SAMPLING_RATE
            camera_path = (
                self.source_data["data_path"] / f"photometry-{self.source_data['session_uuid']}.{camera}.ir.avi"
            )
            camera_paths.append(camera_path)
            camera_timestamps.append(timestamps)

        video_interface = VideoInterface(file_paths=camera_paths, verbose=True)
        video_interface.set_aligned_timestamps(aligned_timestamps=camera_timestamps)
        video_interface.add_to_nwbfile(
            nwbfile=nwbfile,
            metadata=metadata,
            stub_test=False,
            external_mode=True,
            starting_frames=timestamp_offsets,
        )
        video_metadata = dict(
            Behavior=dict(
                Videos=[
                    dict(
                        name="ir_video",
                        description=(
                            "To capture 3D keypoints, mice were recorded in a multi-camera open field arena with "
                            "transparent floor and walls. Near-infrared video recordings at 30 Hz were obtained from "
                            "six cameras (Microsoft Azure Kinect; cameras were placed above, below and at four "
                            "cardinal directions)."
                        ),
                        unit="n.a.",
                    )
                ]
            )
        )
        metadata.update(video_metadata)
