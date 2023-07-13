"""Primary class for converting fiber photometry data (dLight fluorescence)."""
# Standard Scientific Python
import pandas as pd
import numpy as np

# NWB Ecosystem
from pynwb.file import NWBFile
from pynwb.core import DynamicTableRegion
from pynwb.ophys import RoiResponseSeries
from pynwb.ogen import OptogeneticSeries
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.utils import load_dict_from_file
from neuroconv.tools import nwb_helpers
from hdmf.backends.hdf5.h5_utils import H5DataIO

# Local


class OptogeneticInterface(BaseDataInterface):
    """Optogenetic interface for markowitz_gillis_nature_2023 conversion"""

    def __init__(self, file_path: str, session_uuid: str, session_metadata_path: str, subject_metadata_path: str):
        # This should load the data lazily and prepare variables you need
        columns = (
            "uuid",
            "feedback_status",
            "stim_duration",
            "stim_frequency",
            "power",
            "area",
            "timestamp",
        )
        super().__init__(
            file_path=file_path,
            session_uuid=session_uuid,
            columns=columns,
            session_metadata_path=session_metadata_path,
            subject_metadata_path=subject_metadata_path,
        )

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        session_metadata = load_dict_from_file(self.source_data["session_metadata_path"])
        session_metadata = session_metadata[self.source_data["session_uuid"]]
        subject_metadata = load_dict_from_file(self.source_data["subject_metadata_path"])
        subject_metadata = subject_metadata[session_metadata["subject_id"]]

        # Session metadata
        metadata["Optogenetics"]["stim_frequency_Hz"] = session_metadata["stim_frequency_Hz"]
        metadata["Optogenetics"]["pulse_width_s"] = session_metadata["pulse_width_s"]
        metadata["Optogenetics"]["stim_duration_s"] = session_metadata["stim_duration_s"]
        metadata["Optogenetics"]["power_watts"] = session_metadata["power_watts"]

        # Subject metadata
        metadata["Subject"]["sex"] = subject_metadata["sex"]
        metadata["Optogenetics"]["area"] = subject_metadata["optogenetic_area"]

        return metadata

    def get_metadata_schema(self) -> dict:
        metadata_schema = super().get_metadata_schema()
        metadata_schema["properties"]["Optogenetics"] = {
            "type": "object",
            "properties": {
                "area": {"type": "string"},
                "stim_frequency_Hz": {"type": "number"},
                "stim_duration_s": {"type": "number"},
                "power_watts": {"type": "number"},
            },
        }
        return metadata_schema

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        """Run conversion of data from the source file into the nwbfile."""
        session_df = pd.read_parquet(
            self.source_data["file_path"],
            columns=self.source_data["columns"],
            filters=[("uuid", "==", self.source_data["session_uuid"])],
        )

        device = nwbfile.create_device(
            name="OptoEngineMRL",
            description="Optogenetic stimulator (Opto Engine MRL-III-635; SKU: RD-635-00500-CWM-SD-03-LED-0)",
            manufacturer="Opto Engine LLC",
        )
        ogen_site = nwbfile.create_ogen_site(
            name="OptogeneticStimulusSite",
            device=device,
            description="Optogenetic stimulus site",
            excitation_lambda=635.0,
            location=metadata["Optogenetics"]["area"],
        )
        # Reconstruct optogenetic series from feedback status
        stim_duration_s = metadata["Optogenetics"]["stim_duration_s"]
        power_watts = metadata["Optogenetics"]["power_watts"]
        feedback_is_on_index = np.where(session_df.feedback_status == 1)[0]
        data, timestamps = np.zeros(len(feedback_is_on_index) * 2 + 2), np.zeros(len(feedback_is_on_index) * 2 + 2)
        timestamps[0], timestamps[-1] = session_df.timestamp.iloc[0], session_df.timestamp.iloc[-1]
        for i, index in enumerate(feedback_is_on_index):
            t = session_df.timestamp.iloc[index]
            data[i * 2 + 1 : i * 2 + 3] = [power_watts, 0]
            timestamps[i * 2 + 1 : i * 2 + 3] = [t, t + stim_duration_s]
        ogen_series = OptogeneticSeries(
            name="OptogeneticSeries",
            site=ogen_site,
            data=H5DataIO(data, compression=True),
            timestamps=H5DataIO(timestamps, compression=True),
        )
        nwbfile.add_stimulus(ogen_series)

        return nwbfile
