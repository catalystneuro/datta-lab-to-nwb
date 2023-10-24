"""Primary class for converting experiment-specific behavior."""
import numpy as np
import pandas as pd
from pynwb import NWBFile
from .basedattainterface import BaseDattaInterface
from neuroconv.utils import load_dict_from_file
from neuroconv.tools import nwb_helpers
from hdmf.backends.hdf5.h5_utils import H5DataIO
from ndx_events import LabeledEvents


class BehavioralSyllableInterface(BaseDattaInterface):
    """Behavioral Syllable Interface for markowitz_gillis_nature_2023 conversion"""

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
            "predicted_syllable (offline)",
            "predicted_syllable",
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

    def get_metadata_schema(self) -> dict:
        metadata_schema = super().get_metadata_schema()
        metadata_schema["properties"]["BehavioralSyllable"] = {
            "type": "object",
            "properties": {
                "sorted_pseudoindex2name": {"type": "object"},
                "id2sorted_index": {"type": "object"},
                "sorted_index2id": {"type": "object"},
            },
        }
        return metadata_schema

    def get_original_timestamps(self) -> np.ndarray:
        session_df = pd.read_parquet(
            self.source_data["file_path"],
            columns=["timestamp", "uuid"],
            filters=[("uuid", "==", self.source_data["session_uuid"])],
        )
        return session_df["timestamp"].to_numpy()

    def align_timestamps(self, metadata: dict) -> np.ndarray:
        timestamps = self.get_original_timestamps()
        self.set_aligned_timestamps(aligned_timestamps=timestamps)
        if self.source_data["alignment_path"] is not None:
            aligned_starting_time = (
                metadata["Alignment"]["bias"] / metadata["Constants"]["DEMODULATED_PHOTOMETRY_SAMPLING_RATE"]
            )
            self.set_aligned_starting_time(aligned_starting_time=aligned_starting_time)
        return self.aligned_timestamps

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict, velocity_modulation: bool = False) -> None:
        if velocity_modulation:
            columns = ["uuid", "predicted_syllable"]
        else:
            columns = self.source_data["columns"]
        session_df = pd.read_parquet(
            self.source_data["file_path"],
            columns=columns,
            filters=[("uuid", "==", self.source_data["session_uuid"])],
        )
        timestamps = self.align_timestamps(metadata=metadata)
        # Add Syllable Data
        sorted_pseudoindex2name = metadata["BehavioralSyllable"]["sorted_pseudoindex2name"]
        id2sorted_index = metadata["BehavioralSyllable"]["id2sorted_index"]
        syllable_names = np.fromiter(sorted_pseudoindex2name.values(), dtype="O")
        syllable_pseudoindices = np.fromiter(sorted_pseudoindex2name.keys(), dtype=np.int64)
        index2name = syllable_names[np.argsort(syllable_pseudoindices)].tolist()
        for _ in range(len(id2sorted_index) - len(index2name)):
            index2name.append("Uncommon Syllable (frequency < 1%)")
        online_syllable_ids = session_df["predicted_syllable"]
        online_syllable_indices = online_syllable_ids.map(id2sorted_index).to_numpy(dtype=np.uint8)
        online_syllables = LabeledEvents(
            name="BehavioralSyllableOnline",
            description="Behavioral Syllable identified by online Motion Sequencing (MoSeq).",
            timestamps=H5DataIO(timestamps, compression=True),
            data=H5DataIO(online_syllable_indices, compression=True),
            labels=H5DataIO(index2name, compression=True),
        )
        behavior_module = nwb_helpers.get_module(
            nwbfile,
            name="behavior",
            description="Processed behavioral data from MoSeq",
        )
        behavior_module.add(online_syllables)
        if not velocity_modulation:
            offline_syllable_ids = session_df["predicted_syllable (offline)"]
            offline_syllable_indices = offline_syllable_ids.map(id2sorted_index).to_numpy(dtype=np.uint8)
            offline_syllables = LabeledEvents(
                name="BehavioralSyllableOffline",
                description="Behavioral Syllable identified by offline Motion Sequencing (MoSeq).",
                timestamps=H5DataIO(timestamps, compression=True),
                data=H5DataIO(offline_syllable_indices, compression=True),
                labels=H5DataIO(index2name, compression=True),
            )
            behavior_module.add(offline_syllables)
