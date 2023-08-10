"""Primary class for handling velocity-modulation metadata."""
from pynwb import NWBFile
from .basedattainterface import BaseDattaInterface
from neuroconv.utils import load_dict_from_file


class VelocityModulationInterface(BaseDattaInterface):
    """Base interface for markowitz_gillis_nature_2023 conversion w/ velocity-modulation metadata"""

    def __init__(self, file_path: str, session_uuid: str, session_metadata_path: str, subject_metadata_path: str):
        # This should load the data lazily and prepare variables you need
        super().__init__(
            file_path=file_path,
            session_uuid=session_uuid,
            session_metadata_path=session_metadata_path,
            subject_metadata_path=subject_metadata_path,
        )

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        session_metadata = load_dict_from_file(self.source_data["session_metadata_path"])
        session_metadata = session_metadata[self.source_data["session_uuid"]]
        if session_metadata["trigger_syllable_scalar_comparison"] == "lt":
            metadata["NWBFile"]["stimulus_notes"] = "Stim Down: Stimulate when target velocity <25th percentile"
        elif session_metadata["trigger_syllable_scalar_comparison"] == "gt":
            metadata["NWBFile"]["stimulus_notes"] = "Stim Up: Stimulate when target velocity >75th percentile"
        return metadata

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict) -> None:
        pass  # This makes VelocityModulationInterface a non-abtract class
