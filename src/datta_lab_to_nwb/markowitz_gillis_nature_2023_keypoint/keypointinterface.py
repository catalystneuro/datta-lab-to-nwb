"""Primary class for converting experiment-specific behavior."""
import numpy as np
from pynwb import NWBFile
from pynwb.image import RGBImage, Images
import joblib
from ..markowitz_gillis_nature_2023.basedattainterface import BaseDattaInterface
from neuroconv.tools import nwb_helpers
from hdmf.backends.hdf5.h5_utils import H5DataIO
from ndx_pose import PoseEstimationSeries, PoseEstimation


class KeypointInterface(BaseDattaInterface):
    """Keypoint Interface for markowitz_gillis_nature_2023 conversion"""

    def __init__(
        self,
        file_path: str,
        session_uuid: str,
        session_id: str,
        session_metadata_path: str,
        subject_metadata_path: str,
        summary_image_path: str,
    ):
        super().__init__(
            file_path=file_path,
            summary_image_path=summary_image_path,
            session_uuid=session_uuid,
            session_id=session_id,
            session_metadata_path=session_metadata_path,
            subject_metadata_path=subject_metadata_path,
        )

    def get_metadata_schema(self) -> dict:
        metadata_schema = super().get_metadata_schema()
        metadata_schema["properties"]["Keypoint"] = {
            "type": "object",
            "properties": {
                "index_to_name": {"type": "object"},
            },
        }
        return metadata_schema

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict) -> None:
        SAMPLING_RATE = metadata["Constants"]["VIDEO_SAMPLING_RATE"]
        keypoint_dict = joblib.load(self.source_data["file_path"])
        raw_keypoints = keypoint_dict["positions_median"]
        timestamps = H5DataIO(np.arange(raw_keypoints.shape[0]) / SAMPLING_RATE, compression=True)

        index_to_name = metadata["Keypoint"]["index_to_name"]
        camera_names = ["bottom", "side1", "side2", "side3", "side4", "top"]
        keypoints = []
        for camera in camera_names:
            nwbfile.create_device(
                name=f"{camera}_camera", description=f"{camera} IR camera", manufacturer="Microsoft Azure Kinect"
            )
        for keypoint_index, keypoint_name in index_to_name.items():
            keypoint = PoseEstimationSeries(
                name=keypoint_name,
                description=f"Keypoint corresponding to {keypoint_name}",
                data=H5DataIO(raw_keypoints[:, keypoint_index, :], compression=True),
                timestamps=timestamps,
                unit="mm",
                reference_frame=metadata["Keypoint"]["reference_frame"],
            )
            keypoints.append(keypoint)
        keypoints = PoseEstimation(
            pose_estimation_series=keypoints,
            name="keypoints",
            description=(
                "To capture 3D keypoints, mice were recorded in a multi-camera open field arena with "
                "transparent floor and walls. Near-infrared video recordings at 30 Hz were obtained from six "
                "cameras (Microsoft Azure Kinect; cameras were placed above, below and at four cardinal "
                "directions). Separate deep neural networks with an HRNet architecture were trained to detect "
                "keypoints in each view (top, bottom and side) using ~1,000 hand-labelled frames70. "
                "Frame-labelling was crowdsourced through a commercial service (Scale AI), and included the "
                "tail tip, tail base, three points along the spine, the ankle and toe of each hind limb, the "
                "forepaws, ears, nose and implant. After detection of 2D keypoints from each camera, "
                "3D keypoint coordinates were triangulated and then refined using GIMBAL—a model-based "
                "approach that leverages anatomical constraints and motion continuity71. GIMBAL requires "
                "learning an anatomical model and then applying the model to multi-camera behaviour "
                "recordings. For model fitting, we followed the approach described in ref. 71, using 50 pose "
                "states and excluding outlier poses using the EllipticEnvelope method from sklearn. "
                "For applying GIMBAL to behaviour recordings, we again followed71, setting the parameters "
                "obs_outlier_variance, obs_inlier_variance, and pos_dt_variance to 1e6, 10 and 10, "
                "respectively for all keypoints."
            ),
            nodes=list(index_to_name.values()),
            edges=metadata["Keypoint"]["edges"],
        )
        behavior_module = nwb_helpers.get_module(
            nwbfile,
            name="behavior",
            description="3D Keypoints",
        )
        behavior_module.add(keypoints)
        summary_image = joblib.load(self.source_data["summary_image_path"])
        summary_image = RGBImage(
            name="summary_image",
            description="Summary image of 3D keypoints",
            data=summary_image,
        )
        summary_images = Images(name="summary_images", images=[summary_image])
        behavior_module.add(summary_images)
