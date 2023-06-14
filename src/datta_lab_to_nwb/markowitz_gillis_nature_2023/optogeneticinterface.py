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

    def __init__(self, file_path: str, session_uuid: str, metadata_path: str):
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
            metadata_path=metadata_path,
        )

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        return metadata

    def get_metadata_schema(self) -> dict:
        metadata_schema = super().get_metadata_schema()
        return metadata_schema

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        """Run conversion of data from the source file into the nwbfile."""
        session_df = pd.read_parquet(
            self.source_data["file_path"],
            columns=self.source_data["columns"],
            filters=[("uuid", "==", self.source_data["session_uuid"])],
        )
        location = session_df["area"].unique()[0]
        stim_duration = session_df["stim_duration"].unique()[0]
        fs = 30.0
        stim_duration_index = int(stim_duration * fs)
        power_mW = session_df["power"].unique()[0]
        power_W = power_mW / 1000.0

        device = nwbfile.create_device(
            name="Opto Engine MRL-III-635",
            description="Optogenetic stimulator (Opto Engine MRL-III-635; SKU: RD-635-00500-CWM-SD-03-LED-0)",
            manufacturer="Opto Engine LLC",
        )
        ogen_site = nwbfile.create_ogen_site(
            name="OptogeneticStimulusSite",
            device=device,
            description="Optogenetic stimulus site",
            excitation_lambda=635.0,
            location=location,
        )
        # Reconstruct optogenetic series from feedback status
        feedback_status_cts = session_df.feedback_status.to_numpy()
        feedback_is_on_index = np.where(feedback_status_cts == 1)[0]
        for index in feedback_is_on_index:
            feedback_status_cts[index : index + stim_duration_index] = 1
        feedback_status_cts[feedback_status_cts == -1] = 0
        ogen_series = OptogeneticSeries(
            name="OptogeneticSeries",
            site=ogen_site,
            data=H5DataIO(feedback_status_cts * power_W, compression=True),
            timestamps=H5DataIO(session_df.timestamp.to_numpy(), compression=True),
        )
        nwbfile.add_stimulus(ogen_series)

        return nwbfile
