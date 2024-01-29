"""Primary class for converting fiber photometry data (dLight fluorescence)."""

# Standard Scientific Python
import numpy as np
import joblib

# NWB Ecosystem
from pynwb.file import NWBFile
from pynwb.ophys import RoiResponseSeries
from ..markowitz_gillis_nature_2023.rawfiberphotometryinterface import RawFiberPhotometryInterface
from neuroconv.tools import nwb_helpers
from hdmf.backends.hdf5.h5_utils import H5DataIO


class FiberPhotometryInterface(RawFiberPhotometryInterface):
    def __init__(
        self,
        file_path: str,
        tdt_path: str,
        tdt_metadata_path: str,
        session_uuid: str,
        session_id: str,
        session_metadata_path: str,
        subject_metadata_path: str,
    ):
        super().__init__(
            file_path=file_path,
            tdt_path=tdt_path,
            tdt_metadata_path=tdt_metadata_path,
            session_uuid=session_uuid,
            session_id=session_id,
            session_metadata_path=session_metadata_path,
            subject_metadata_path=subject_metadata_path,
        )

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict) -> None:
        SAMPLING_RATE = 30
        super().add_to_nwbfile(nwbfile, metadata)
        processed_photometry = joblib.load(self.source_data["file_path"])
        timestamps = np.arange(processed_photometry["dlight"].shape[0]) / SAMPLING_RATE
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
