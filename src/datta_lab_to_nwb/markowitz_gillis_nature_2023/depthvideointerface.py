"""Primary class for converting raw depth video data."""
from pynwb import NWBFile
from .basevideointerface import BaseVideoInterface


class DepthVideoInterface(BaseVideoInterface):
    """Depth video interface for markowitz_gillis_nature_2023 conversion"""

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict) -> None:
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
        metadata.update(video_metadata)
        super().add_to_nwbfile(nwbfile, metadata)
