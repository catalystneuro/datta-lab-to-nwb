"""Primary class for handling metadata non-specific to any other DataInterfaces."""
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.utils import load_dict_from_file
from pynwb import NWBFile


class MetadataInterface(BaseDataInterface):
    """Metadata interface for markowitz_gillis_nature_2023 conversion"""

    def __init__(self, session_uuid: str, session_metadata_path: str, subject_metadata_path: str):
        super().__init__(
            session_uuid=session_uuid,
            session_metadata_path=session_metadata_path,
            subject_metadata_path=subject_metadata_path,
        )

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        session_metadata = load_dict_from_file(self.source_data["session_metadata_path"])
        session_metadata = session_metadata[self.source_data["session_uuid"]]
        subject_metadata = load_dict_from_file(self.source_data["subject_metadata_path"])
        subject_metadata = subject_metadata[session_metadata["subject_id"]]

        metadata["NWBFile"]["session_description"] = session_metadata["session_description"]
        metadata["NWBFile"]["session_start_time"] = session_metadata["session_start_time"]
        metadata["NWBFile"]["identifier"] = self.source_data["session_uuid"]
        metadata["NWBFile"]["session_id"] = self.source_data["session_uuid"]

        metadata["Subject"] = {}
        metadata["Subject"]["subject_id"] = session_metadata["subject_id"]
        metadata["Subject"]["sex"] = subject_metadata["sex"]

        return metadata

    def run_conversion(self, nwbfile: NWBFile, metadata: dict) -> NWBFile:
        """Run conversion of data from the source file into the nwbfile."""
        return nwbfile
