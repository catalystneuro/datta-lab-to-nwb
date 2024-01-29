"""Primary class for converting fiber photometry data (dLight fluorescence)."""

# Standard Scientific Python
import pandas as pd
import numpy as np

# NWB Ecosystem
from pynwb.file import NWBFile
from pynwb.core import DynamicTableRegion
from pynwb.ophys import RoiResponseSeries
from pynwb.ogen import OptogeneticSeries
from .basedattainterface import BaseDattaInterface
from neuroconv.utils import load_dict_from_file
from neuroconv.tools import nwb_helpers
from hdmf.backends.hdf5.h5_utils import H5DataIO

# Local


class OptogeneticInterface(BaseDattaInterface):
    """Optogenetic interface for markowitz_gillis_nature_2023 conversion"""

    def __init__(
        self,
        file_path: str,
        session_uuid: str,
        session_id: str,
        session_metadata_path: str,
        subject_metadata_path: str,
        alignment_path: str = None,
    ):
        # This should load the data lazily and prepare variables you need
        columns = (
            "uuid",
            "feedback_status",
            "timestamp",
        )
        super().__init__(
            file_path=file_path,
            session_uuid=session_uuid,
            session_id=session_id,
            columns=columns,
            session_metadata_path=session_metadata_path,
            subject_metadata_path=subject_metadata_path,
            alignment_path=alignment_path,
        )

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        session_metadata = load_dict_from_file(self.source_data["session_metadata_path"])
        session_metadata = session_metadata[self.source_data["session_uuid"]]
        subject_metadata = load_dict_from_file(self.source_data["subject_metadata_path"])
        subject_metadata = subject_metadata[session_metadata["subject_id"]]

        metadata["Optogenetics"]["stim_frequency_Hz"] = session_metadata["stim_frequency_Hz"]
        metadata["Optogenetics"]["pulse_width_s"] = session_metadata["pulse_width_s"]
        metadata["Optogenetics"]["stim_duration_s"] = session_metadata["stim_duration_s"]
        metadata["Optogenetics"]["power_watts"] = session_metadata["power_watts"]
        metadata["Optogenetics"]["area"] = subject_metadata["optogenetic_area"]
        metadata["Optogenetics"]["target_syllable"] = session_metadata["target_syllable"]
        if "velocity_modulation" in session_metadata.keys():
            if session_metadata["trigger_syllable_scalar_comparison"] == "lt":
                metadata["NWBFile"]["stimulus_notes"] = "Stim Down: Stimulate when target velocity <25th percentile"
            elif session_metadata["trigger_syllable_scalar_comparison"] == "gt":
                metadata["NWBFile"]["stimulus_notes"] = "Stim Up: Stimulate when target velocity >75th percentile"

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
                "pulse_width_s": {"type": "number"},
                "target_syllable": {"type": "array"},
            },
        }
        return metadata_schema

    def get_original_timestamps(self, metadata: dict) -> np.ndarray:
        session_df = pd.read_parquet(
            self.source_data["file_path"],
            columns=["timestamp", "uuid"],
            filters=[
                ("uuid", "==", self.source_data["session_uuid"]),
                ("target_syllable", "==", metadata["Optogenetics"]["target_syllable"][0]),
            ],
        )
        return session_df["timestamp"].to_numpy()

    def align_timestamps(self, metadata: dict, velocity_modulation: bool) -> np.ndarray:
        timestamps = self.get_original_timestamps(metadata=metadata)
        self.set_aligned_timestamps(aligned_timestamps=timestamps)
        if self.source_data["alignment_path"] is not None:
            aligned_starting_time = (
                metadata["Alignment"]["bias"] / metadata["Constants"]["DEMODULATED_PHOTOMETRY_SAMPLING_RATE"]
            )
            self.set_aligned_starting_time(aligned_starting_time=aligned_starting_time)
        elif velocity_modulation:
            aligned_starting_time = 2700  # See Methods: Closed-loop velocity modulation experiments
            self.set_aligned_starting_time(aligned_starting_time=aligned_starting_time)
        return self.aligned_timestamps

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict, velocity_modulation: bool = False) -> None:
        session_timestamps = self.align_timestamps(metadata=metadata, velocity_modulation=velocity_modulation)
        session_df = pd.read_parquet(
            self.source_data["file_path"],
            columns=self.source_data["columns"],
            filters=[
                ("uuid", "==", self.source_data["session_uuid"]),
                ("target_syllable", "==", metadata["Optogenetics"]["target_syllable"][0]),
            ],
        )

        # Reconstruct optogenetic series from feedback status
        if pd.isnull(metadata["Optogenetics"]["stim_frequency_Hz"]):  # cts stim
            data, timestamps = self.reconstruct_cts_stim(metadata, session_df, session_timestamps)
        else:  # pulsed stim
            data, timestamps = self.reconstruct_pulsed_stim(metadata, session_df, session_timestamps)

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
        id2sorted_index = metadata["BehavioralSyllable"]["id2sorted_index"]
        target_syllables = [id2sorted_index[syllable_id] for syllable_id in metadata["Optogenetics"]["target_syllable"]]
        ogen_series = OptogeneticSeries(
            name="OptogeneticSeries",
            description="Onset of optogenetic stimulation is recorded as a 1, and offset is recorded as a 0.",
            comments=f"target_syllable(s) = {target_syllables}",
            site=ogen_site,
            data=H5DataIO(data, compression=True),
            timestamps=H5DataIO(timestamps, compression=True),
        )
        nwbfile.add_stimulus(ogen_series)

        return nwbfile

    def reconstruct_cts_stim(self, metadata, session_df, session_timestamps):
        stim_duration_s = metadata["Optogenetics"]["stim_duration_s"]
        power_watts = metadata["Optogenetics"]["power_watts"]
        feedback_is_on_index = np.where(session_df.feedback_status == 1)[0]
        data_len = len(feedback_is_on_index) * 2 + 2
        data, timestamps = np.zeros(data_len), np.zeros(data_len)
        timestamps[0], timestamps[-1] = session_timestamps[0], session_timestamps[-1]
        for i, index in enumerate(feedback_is_on_index):
            t = session_timestamps[index]
            data[i * 2 + 1 : i * 2 + 3] = [power_watts, 0]
            timestamps[i * 2 + 1 : i * 2 + 3] = [t, t + stim_duration_s]
        sorting_index = np.argsort(timestamps)
        data, timestamps = data[sorting_index], timestamps[sorting_index]
        return data, timestamps

    def reconstruct_pulsed_stim(self, metadata, session_df, session_timestamps):
        stim_duration_s = metadata["Optogenetics"]["stim_duration_s"]
        power_watts = metadata["Optogenetics"]["power_watts"]
        stim_frequency_Hz = metadata["Optogenetics"]["stim_frequency_Hz"]
        pulse_width_s = metadata["Optogenetics"]["pulse_width_s"]
        feedback_is_on_index = np.where(session_df.feedback_status == 1)[0]
        pulses_per_stim = int(stim_duration_s * stim_frequency_Hz)
        data_len = len(feedback_is_on_index) * 2 * pulses_per_stim + 2
        data, timestamps = np.zeros(data_len), np.zeros(data_len)
        timestamps[0], timestamps[-1] = session_timestamps[0], session_timestamps[-1]
        for i, index in enumerate(feedback_is_on_index):
            t0 = session_timestamps[index]
            for pulse in range(pulses_per_stim):
                t_on = t0 + pulse * 1 / stim_frequency_Hz
                t_off = t_on + pulse_width_s
                data_index = i * 2 * pulses_per_stim + 2 * pulse + 1
                data[data_index : data_index + 2] = [power_watts, 0]
                timestamps[data_index : data_index + 2] = [t_on, t_off]
        sorting_index = np.argsort(timestamps)
        data, timestamps = data[sorting_index], timestamps[sorting_index]
        return data, timestamps
