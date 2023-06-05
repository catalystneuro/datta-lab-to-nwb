"""Primary class for converting fiber photometry data (dLight fluorescence)."""
# Standard Scientific Python
import pandas as pd

# NWB Ecosystem
from pynwb.file import NWBFile
from pynwb.core import DynamicTableRegion
from pynwb.ophys import RoiResponseSeries
from ndx_photometry import (
    FibersTable,
    PhotodetectorsTable,
    ExcitationSourcesTable,
    DeconvolvedRoiResponseSeries,
    MultiCommandedVoltage,
    FiberPhotometry,
    FluorophoresTable,
)
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.utils import load_dict_from_file
from neuroconv.tools import nwb_helpers
from hdmf.backends.hdf5.h5_utils import H5DataIO

# Local


class FiberPhotometryInterface(BaseDataInterface):
    """Fiber Photometry  interface for markowitz_gillis_nature_2023 conversion"""

    def __init__(self, file_path: str, session_uuid: str, metadata_path: str):
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
            session_uuid=session_uuid,
            columns=columns,
            metadata_path=metadata_path,
        )

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        session_metadata = load_dict_from_file(self.source_data["metadata_path"])
        session_metadata = session_metadata[self.source_data["session_uuid"]]
        metadata["NWBFile"]["session_description"] = session_metadata["session_description"]
        metadata["NWBFile"]["session_start_time"] = session_metadata["session_start_time"]
        metadata["NWBFile"]["identifier"] = self.source_data["session_uuid"]
        metadata["NWBFile"]["session_id"] = self.source_data["session_uuid"]
        metadata["Subject"] = {}
        metadata["Subject"]["subject_id"] = session_metadata["subject_id"]
        metadata["FiberPhotometry"] = {}
        metadata["FiberPhotometry"]["area"] = session_metadata["area"]
        metadata["FiberPhotometry"]["reference_max"] = session_metadata["reference_max"]
        metadata["FiberPhotometry"]["signal_max"] = session_metadata["signal_max"]
        metadata["FiberPhotometry"]["signal_reference_corr"] = session_metadata["signal_reference_corr"]
        metadata["FiberPhotometry"]["snr"] = session_metadata["snr"]

        return metadata

    def get_metadata_schema(self) -> dict:
        metadata_schema = super().get_metadata_schema()
        metadata_schema["properties"]["FiberPhotometry"] = {
            "type": "object",
            "properties": {
                "area": {"type": "string"},
                "reference_max": {"type": "number"},
                "signal_max": {"type": "number"},
                "signal_reference_corr": {"type": "number"},
                "snr": {"type": "number"},
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

        # Excitation Sources Table
        excitation_sources_table = ExcitationSourcesTable(
            description=(
                "A 470nm (blue) LED and a 405nM (UV) LED (Mightex) were sinusoidally modulated at "
                "161Hz and 381Hz, respectively (these frequencies were chosen to avoid harmonic cross-talk). "
                "Modulated excitation light was passed through a three-colour fluorescence mini-cube "
                "(Doric Lenses FMC7_E1(400-410)_F1(420-450)_E2(460-490)_F2(500-540)_E3(550-575)_F3(600-680)_S), "
                "then through a pigtailed rotary joint "
                "(Doric Lenses B300-0089, FRJ_1x1_PT_200/220/LWMJ-0.37_1.0m_FCM_0.08m_FCM) and finally into a "
                "low-autofluorescence fibre-optic patch cord "
                "(Doric Lenses MFP_200/230/900-0.37_0.75m_FCM-MF1.25_LAF or MFP_200/230/900-0.57_0.75m_FCM-MF1.25_LAF) "
                "connected to the optical implant in the freely moving mouse."
            )
        )
        excitation_sources_table.add_row(
            peak_wavelength=470.0,
            source_type="laser",
        )
        excitation_sources_table.add_row(
            peak_wavelength=405.0,
            source_type="laser",
        )

        # Photodetectors Table
        photodetectors_table = PhotodetectorsTable(
            description=(
                "Emission light was collected through the same patch cord, then passed back through the mini-cube. "
                "Light on the F2 port was bandpass filtered for green emission (500–540nm) and sent to a silicon "
                "photomultiplier with an integrated transimpedance amplifier (SensL MiniSM-30035-X08). Voltages from "
                "the SensL unit were collected through the TDT Active X interface using 24-bit analogue-to-digital "
                "convertors at >6kHz, and voltage signals driving the UV and blue LEDs were also stored for "
                "offline analysis."
            ),
        )
        photodetectors_table.add_row(peak_wavelength=527.0, type="PMT", gain=1.0)

        # Fluorophores Table
        fluorophores_table = FluorophoresTable(
            description=(
                "dLight1.1 was selected to visualize dopamine release dynamics in the DLS owing to its rapid rise and "
                "decay times, comparatively lower dopamine affinity (so as to not saturate binding), as well as its "
                "responsiveness over much of the physiological range of known DA concentrations in freely moving "
                "rodents."
            ),
        )

        fluorophores_table.add_row(
            label="dlight1.1",
            location=metadata["FiberPhotometry"]["area"],
            coordinates=(0.260, 2.550, -2.40),  # (AP, ML, DV)
        )

        # Fibers Table
        fibers_table = FibersTable(
            description=(
                "Fiber photometry data with 2 excitation sources (470nm and 405nm), 1 PMT photodetector with "
                "a peak wavelength of 527nm, and 1 fluorophore (dLight1.1)."
            )
        )
        nwbfile.add_lab_meta_data(
            FiberPhotometry(
                fibers=fibers_table,
                excitation_sources=excitation_sources_table,
                photodetectors=photodetectors_table,
                fluorophores=fluorophores_table,
            )
        )

        # Important: we add the fibers to the fibers table _after_ adding the metadata
        # This ensures that we can find this data in their tables of origin
        fibers_table.add_fiber(
            excitation_source=0,  # integers indicated rows of excitation sources table
            photodetector=0,
            fluorophores=[0],  # potentially multiple fluorophores, so list of indices
            location=metadata["FiberPhotometry"]["area"],
            notes="None",
        )
        fibers_table.add_fiber(
            excitation_source=1,  # integers indicated rows of excitation sources table
            photodetector=0,
            fluorophores=[0],  # potentially multiple fluorophores, so list of indices
            location=metadata["FiberPhotometry"]["area"],
            notes="None",
        )

        # ROI Response Series
        # Here we set up a list of fibers that our recording came from
        fibers_ref = DynamicTableRegion(name="rois", data=[0, 1], description="source fibers", table=fibers_table)
        signal_series = RoiResponseSeries(
            name="SignalDfOverF",
            description="The ΔF/F from the blue light excitation (480nm) corresponding to the dopamine signal.",
            data=H5DataIO(session_df.signal_dff.to_numpy(), compression=True),
            unit="a.u.",
            timestamps=H5DataIO(session_df.timestamp.to_numpy(), compression=True),
            rois=fibers_ref,
        )
        reference_series = RoiResponseSeries(
            name="ReferenceDfOverF",
            description="The ∆F/F from the isosbestic UV excitation (400nm) corresponding to the reference signal.",
            data=H5DataIO(session_df.reference_dff.to_numpy(), compression=True),
            unit="a.u.",
            timestamps=signal_series.timestamps,
            rois=fibers_ref,
        )
        reference_fit_series = RoiResponseSeries(
            name="ReferenceDfOverFSmoothed",
            description=(
                "The ∆F/F from the isosbestic UV excitation (400nm) that has been smoothed "
                "(See Methods: Photometry Active Referencing)."
            ),
            data=H5DataIO(session_df.reference_dff_fit.to_numpy(), compression=True),
            unit="dF/F",
            timestamps=signal_series.timestamps,
            rois=fibers_ref,
        )
        uv_reference_fit_series = RoiResponseSeries(
            name="UVReferenceFSmoothed",
            description=(
                "Raw F from the isosbestic UV excitation (400nm) that has been smoothed "
                "(See Methods: Photometry Active Referencing)."
            ),
            data=H5DataIO(session_df.uv_reference_fit.to_numpy(), compression=True),
            unit="F",
            timestamps=signal_series.timestamps,
            rois=fibers_ref,
        )

        # Aggregate into OPhys Module
        ophys_module = nwb_helpers.get_module(nwbfile, name="ophys", description="Fiber photometry data")
        ophys_module.add(signal_series)
        ophys_module.add(reference_series)
        ophys_module.add(reference_fit_series)
        ophys_module.add(uv_reference_fit_series)

        return nwbfile
