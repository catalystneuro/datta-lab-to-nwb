"""Primary class for converting fiber photometry data (dLight fluorescence)."""
# Standard Scientific Python
import numpy as np
import joblib

# NWB Ecosystem
from pynwb.file import NWBFile
from pynwb.ophys import RoiResponseSeries
from ..markowitz_gillis_nature_2023.rawfiberphotometryinterface import RawFiberPhotometryInterface, load_tdt_data
from neuroconv.tools import nwb_helpers
from hdmf.backends.hdf5.h5_utils import H5DataIO


class FiberPhotometryInterface(RawFiberPhotometryInterface):
    def __init__(
        self,
        file_path: str,
        tdt_path: str,
        tdt_metadata_path: str,
        depth_timestamp_path: str,
        session_uuid: str,
        session_id: str,
        session_metadata_path: str,
        subject_metadata_path: str,
        alignment_path: str = None,
    ):
        super().__init__(
            file_path=file_path,
            tdt_path=tdt_path,
            tdt_metadata_path=tdt_metadata_path,
            depth_timestamp_path=depth_timestamp_path,
            session_uuid=session_uuid,
            session_id=session_id,
            session_metadata_path=session_metadata_path,
            subject_metadata_path=subject_metadata_path,
            alignment_path=alignment_path,
        )

    def get_original_timestamps(self, metadata) -> np.ndarray:
        processed_photometry = joblib.load(self.source_data["file_path"])
        timestamps = np.arange(processed_photometry["dlight"].shape[0]) / metadata["Constants"]["VIDEO_SAMPLING_RATE"]
        return timestamps

    def align_processed_timestamps(
        self, metadata: dict
    ) -> np.ndarray:  # TODO: align timestamps if we get alignment_df.parquet
        timestamps = self.get_original_timestamps(metadata=metadata)
        self.set_aligned_timestamps(aligned_timestamps=timestamps)
        return self.aligned_timestamps

    def align_raw_timestamps(self, metadata: dict) -> np.ndarray:  # TODO: remove if we get alignment_df.parquet
        photometry_dict = load_tdt_data(self.source_data["tdt_path"], fs=metadata["FiberPhotometry"]["raw_rate"])
        timestamps = photometry_dict["tstep"]
        self.set_aligned_timestamps(aligned_timestamps=timestamps)
        return self.aligned_timestamps

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict) -> None:
        super().add_to_nwbfile(nwbfile, metadata)
        processed_photometry = joblib.load(self.source_data["file_path"])
        timestamps = self.align_processed_timestamps(metadata)
        signal_series = RoiResponseSeries(
            name="SignalF",
            description=(
                "Demodulated raw fluorescence (F) from the blue light excitation (470nm) "
                "(See Methods: Photometry Active Referencing)."
            ),
            data=H5DataIO(processed_photometry["dlight"], compression=True),
            unit="n.a.",
            timestamps=H5DataIO(timestamps, compression=True),
            rois=self.fibers_ref,
        )
        reference_series = RoiResponseSeries(
            name="UVReferenceF",
            description=(
                "Demodulated raw fluorescence (F) from the isosbestic UV excitation (405nm) "
                "(See Methods: Photometry Active Referencing)."
            ),
            data=H5DataIO(processed_photometry["uv"], compression=True),
            unit="n.a.",
            timestamps=signal_series.timestamps,
            rois=self.fibers_ref,
        )
        ophys_module = nwb_helpers.get_module(nwbfile, name="ophys", description="Fiber photometry data")
        ophys_module.add(signal_series)
        ophys_module.add(reference_series)
