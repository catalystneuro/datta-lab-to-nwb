"""Primary class for converting fiber photometry data (dLight fluorescence)."""
# Standard Scientific Python
import pandas as pd
import numpy as np

# NWB Ecosystem
from pynwb.file import NWBFile
from pynwb.ophys import RoiResponseSeries
from .rawfiberphotometryinterface import RawFiberPhotometryInterface
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
        # This should load the data lazily and prepare variables you need
        columns = (
            "uuid",
            "signal_dff",
            "reference_dff",
            "uv_reference_fit",
            "reference_dff_fit",
        )
        super().__init__(
            file_path=file_path,
            tdt_path=tdt_path,
            tdt_metadata_path=tdt_metadata_path,
            depth_timestamp_path=depth_timestamp_path,
            session_uuid=session_uuid,
            session_id=session_id,
            columns=columns,
            session_metadata_path=session_metadata_path,
            subject_metadata_path=subject_metadata_path,
            alignment_path=alignment_path,
        )

    def get_original_timestamps(self) -> np.ndarray:
        session_df = pd.read_parquet(
            self.source_data["file_path"],
            columns=["timestamp", "uuid"],
            filters=[("uuid", "==", self.source_data["session_uuid"])],
        )
        return session_df["timestamp"].to_numpy()

    def align_processed_timestamps(self, metadata: dict) -> np.ndarray:
        timestamps = self.get_original_timestamps()
        self.set_aligned_timestamps(aligned_timestamps=timestamps)
        if self.source_data["alignment_path"] is not None:
            aligned_starting_time = (
                metadata["Alignment"]["bias"] / metadata["Constants"]["DEMODULATED_PHOTOMETRY_SAMPLING_RATE"]
            )
            self.set_aligned_starting_time(aligned_starting_time=aligned_starting_time)
        return self.aligned_timestamps

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict) -> None:
        super().add_to_nwbfile(nwbfile, metadata)
        timestamps = self.align_processed_timestamps(metadata)
        session_df = pd.read_parquet(
            self.source_data["file_path"],
            columns=self.source_data["columns"],
            filters=[("uuid", "==", self.source_data["session_uuid"])],
        )
        notnan = pd.notnull(session_df.signal_dff)
        signal_series = RoiResponseSeries(
            name="SignalDfOverF",
            description="The ΔF/F from the blue light excitation (470nm) corresponding to the dopamine signal.",
            data=H5DataIO(session_df.signal_dff.to_numpy()[notnan], compression=True),
            unit="a.u.",
            timestamps=H5DataIO(timestamps[notnan], compression=True),
            rois=self.fibers_ref,
        )
        reference_series = RoiResponseSeries(
            name="ReferenceDfOverF",
            description="The ∆F/F from the isosbestic UV excitation (405nm) corresponding to the reference signal.",
            data=H5DataIO(session_df.reference_dff.to_numpy()[notnan], compression=True),
            unit="a.u.",
            timestamps=signal_series.timestamps,
            rois=self.fibers_ref,
        )
        reference_fit_series = RoiResponseSeries(
            name="ReferenceDfOverFSmoothed",
            description=(
                "The ∆F/F from the isosbestic UV excitation (405nm) that has been smoothed "
                "(See Methods: Photometry Active Referencing)."
            ),
            data=H5DataIO(session_df.reference_dff_fit.to_numpy()[notnan], compression=True),
            unit="a.u.",
            timestamps=signal_series.timestamps,
            rois=self.fibers_ref,
        )
        uv_reference_fit_series = RoiResponseSeries(
            name="UVReferenceFSmoothed",
            description=(
                "Raw fluorescence (F) from the isosbestic UV excitation (405nm) that has been smoothed "
                "(See Methods: Photometry Active Referencing)."
            ),
            data=H5DataIO(session_df.uv_reference_fit.to_numpy()[notnan], compression=True),
            unit="n.a.",
            timestamps=signal_series.timestamps,
            rois=self.fibers_ref,
        )
        ophys_module = nwb_helpers.get_module(nwbfile, name="ophys", description="Fiber photometry data")
        ophys_module.add(signal_series)
        ophys_module.add(reference_series)
        ophys_module.add(reference_fit_series)
        ophys_module.add(uv_reference_fit_series)
