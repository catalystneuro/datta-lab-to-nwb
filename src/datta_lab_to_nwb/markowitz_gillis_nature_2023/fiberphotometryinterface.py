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
        session_uuid: str,
        session_id: str,
        session_metadata_path: str,
        subject_metadata_path: str,
    ):
        # This should load the data lazily and prepare variables you need
        columns = (
            "uuid",
            "signal_dff",
            "reference_dff",
            "uv_reference_fit",
            "reference_dff_fit",
            "timestamp",
        )
        super().__init__(
            file_path=file_path,
            tdt_path=tdt_path,
            tdt_metadata_path=tdt_metadata_path,
            session_uuid=session_uuid,
            session_id=session_id,
            columns=columns,
            session_metadata_path=session_metadata_path,
            subject_metadata_path=subject_metadata_path,
        )

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict) -> None:
        super().add_to_nwbfile(nwbfile, metadata)
        session_df = pd.read_parquet(
            self.source_data["file_path"],
            columns=self.source_data["columns"],
            filters=[("uuid", "==", self.source_data["session_uuid"])],
        )
        signal_series = RoiResponseSeries(
            name="SignalDfOverF",
            description="The ΔF/F from the blue light excitation (470nm) corresponding to the dopamine signal.",
            data=H5DataIO(session_df.signal_dff.to_numpy(), compression=True),
            unit="a.u.",
            timestamps=H5DataIO(session_df.timestamp.to_numpy(), compression=True),
            rois=self.fibers_ref,
        )
        reference_series = RoiResponseSeries(
            name="ReferenceDfOverF",
            description="The ∆F/F from the isosbestic UV excitation (405nm) corresponding to the reference signal.",
            data=H5DataIO(session_df.reference_dff.to_numpy(), compression=True),
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
            data=H5DataIO(session_df.reference_dff_fit.to_numpy(), compression=True),
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
            data=H5DataIO(session_df.uv_reference_fit.to_numpy(), compression=True),
            unit="n.a.",
            timestamps=signal_series.timestamps,
            rois=self.fibers_ref,
        )
        ophys_module = nwb_helpers.get_module(nwbfile, name="ophys", description="Fiber photometry data")
        ophys_module.add(signal_series)
        ophys_module.add(reference_series)
        ophys_module.add(reference_fit_series)
        ophys_module.add(uv_reference_fit_series)


def load_tdt_data(filename, pmt_channels=[0, 3], sync_channel=6, clock_channel=7, nch=8, fs=6103.515625):
    float_data = np.fromfile(filename, dtype=">f4")
    int_data = np.fromfile(filename, dtype=">i4")

    photometry_dict = {}

    for i, pmt in enumerate(pmt_channels):
        photometry_dict["pmt{:02d}".format(i)] = float_data[pmt::nch]
        photometry_dict["pmt{:02d}_x".format(i)] = float_data[pmt + 1 :: nch]
        photometry_dict["pmt{:02d}_y".format(i)] = float_data[pmt + 2 :: nch]

    photometry_dict["sync"] = int_data[sync_channel::8]
    photometry_dict["clock"] = int_data[clock_channel::8]

    if any(np.diff(photometry_dict["clock"]) != 1):
        raise IOError("Timebase not uniform in TDT file.")

    clock_df = np.diff(photometry_dict["clock"].astype("float64"))
    clock_df = np.insert(clock_df, 0, 0, axis=0)
    photometry_dict["tstep"] = np.cumsum(clock_df * 1 / fs)

    return photometry_dict
