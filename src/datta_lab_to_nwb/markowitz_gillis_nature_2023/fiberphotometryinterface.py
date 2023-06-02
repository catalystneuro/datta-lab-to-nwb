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
            description="ExcitationSourcesTable",
        )
        excitation_sources_table.add_row(
            peak_wavelength=480.0,
            source_type="laser",
        )
        excitation_sources_table.add_row(
            peak_wavelength=400.0,
            source_type="laser",
        )

        # Photodetectors Table
        photodetectors_table = PhotodetectorsTable(
            description="PhotodetectorsTable",
        )
        photodetectors_table.add_row(peak_wavelength=527.0, type="PMT", gain=1.0)

        # Fluorophores Table
        fluorophores_table = FluorophoresTable(description="fluorophores")

        fluorophores_table.add_row(
            label="dlight1.1",
            location=metadata["FiberPhotometry"]["area"],
            coordinates=(0.260, 2.550, -2.40),  # (AP, ML, DV)
        )

        # Fibers Table
        fibers_table = FibersTable(
            description="FibersTable",
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
        fibers_ref = DynamicTableRegion(
            name="rois", data=[0, 1], description="source fibers", table=fibers_table  # potentially multiple fibers
        )

        signal_series = RoiResponseSeries(
            name="SignalDfOverF",
            description="signal dF over F",
            data=session_df.signal_dff.to_numpy(),
            unit="a.u.",
            timestamps=session_df.timestamp.to_numpy(),
            rois=fibers_ref,
        )
        reference_series = RoiResponseSeries(
            name="reference_dff",
            description="reference dF over F",
            data=session_df.reference_dff.to_numpy(),
            unit="dF/F",
            timestamps=signal_series.timestamps,
            rois=fibers_ref,
        )
        reference_fit_series = RoiResponseSeries(
            name="reference_dff_fit",
            description="reference fit dF over F",
            data=session_df.reference_dff_fit.to_numpy(),
            unit="dF/F",
            timestamps=signal_series.timestamps,
            rois=fibers_ref,
        )
        uv_reference_fit_series = RoiResponseSeries(
            name="uv_reference_fit",
            description="uv reference F",
            data=session_df.uv_reference_fit.to_numpy(),
            unit="F",
            timestamps=signal_series.timestamps,
            rois=fibers_ref,
        )

        # Aggregate into OPhys Module
        ophys_module = nwbfile.create_processing_module(name="ophys", description="fiber photometry")
        ophys_module.add(signal_series)
        ophys_module.add(reference_series)
        ophys_module.add(reference_fit_series)
        ophys_module.add(uv_reference_fit_series)

        return nwbfile
