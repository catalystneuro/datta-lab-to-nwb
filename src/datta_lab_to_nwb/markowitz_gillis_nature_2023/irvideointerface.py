"""Primary class for converting raw ir video data."""

from pynwb import NWBFile
from .basevideointerface import BaseVideoInterface


class IRVideoInterface(BaseVideoInterface):
    """IR video interface for markowitz_gillis_nature_2023 conversion"""

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict) -> None:
        video_metadata = dict(
            Behavior=dict(
                Videos=[
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
                ]
            )
        )
        metadata.update(video_metadata)
        super().add_to_nwbfile(nwbfile, metadata)
