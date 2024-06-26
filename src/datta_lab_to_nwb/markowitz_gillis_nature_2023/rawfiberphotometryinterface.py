"""Primary class for converting Raw fiber photometry data (dLight fluorescence) from the TDT system."""

from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pynwb.file import NWBFile
from pynwb.core import DynamicTableRegion
from pynwb.ophys import RoiResponseSeries
from ndx_photometry import (
    FibersTable,
    PhotodetectorsTable,
    ExcitationSourcesTable,
    MultiCommandedVoltage,
    FiberPhotometry,
    FluorophoresTable,
)
from neuroconv.utils import load_dict_from_file, FilePathType
from neuroconv.tools import nwb_helpers
from hdmf.backends.hdf5.h5_utils import H5DataIO

from .basedattainterface import BaseDattaInterface
from .utils import convert_timestamps_to_seconds


class RawFiberPhotometryInterface(BaseDattaInterface):
    """Raw Fiber Photometry interface for markowitz_gillis_nature_2023 conversion."""

    def __init__(
        self,
        depth_timestamp_path: str,
        session_uuid: str,
        session_id: str,
        session_metadata_path: str,
        subject_metadata_path: str,
        tdt_path: Union[FilePathType, None] = None,
        tdt_metadata_path: Union[FilePathType, None] = None,
        alignment_path: str = None,
        **kwargs,
    ):
        super().__init__(
            tdt_path=tdt_path,
            tdt_metadata_path=tdt_metadata_path,
            session_uuid=session_uuid,
            session_id=session_id,
            session_metadata_path=session_metadata_path,
            subject_metadata_path=subject_metadata_path,
            alignment_path=alignment_path,
            depth_timestamp_path=depth_timestamp_path,
            **kwargs,
        )

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        session_metadata = load_dict_from_file(self.source_data["session_metadata_path"])
        subject_metadata = load_dict_from_file(self.source_data["subject_metadata_path"])
        session_metadata = session_metadata[self.source_data["session_uuid"]]
        subject_metadata = subject_metadata[session_metadata["subject_id"]]

        metadata["FiberPhotometry"]["reference_max"] = session_metadata["reference_max"]
        metadata["FiberPhotometry"]["signal_max"] = session_metadata["signal_max"]
        metadata["FiberPhotometry"]["signal_reference_corr"] = session_metadata["signal_reference_corr"]
        metadata["FiberPhotometry"]["snr"] = session_metadata["snr"]
        metadata["FiberPhotometry"]["area"] = subject_metadata["photometry_area"]

        if self.source_data["tdt_metadata_path"] is not None:
            tdt_metadata = load_dict_from_file(self.source_data["tdt_metadata_path"])
            metadata["FiberPhotometry"]["gain"] = float(tdt_metadata["tags"]["OutputGain"])
            metadata["FiberPhotometry"]["signal_amp"] = tdt_metadata["tags"]["LED1Amp"]
            metadata["FiberPhotometry"]["reference_amp"] = tdt_metadata["tags"]["LED2Amp"]
            metadata["FiberPhotometry"]["signal_freq"] = float(tdt_metadata["tags"]["LED1Freq"])
            metadata["FiberPhotometry"]["reference_freq"] = float(tdt_metadata["tags"]["LED2Freq"])
            metadata["FiberPhotometry"]["raw_rate"] = tdt_metadata["status"]["sampling_rate"]

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

    def align_raw_timestamps(self, metadata: dict) -> np.ndarray:
        photometry_dict = load_tdt_data(self.source_data["tdt_path"], fs=metadata["FiberPhotometry"]["raw_rate"])
        timestamps = photometry_dict["tstep"]
        depth_timestamps = pd.read_csv(self.source_data["depth_timestamp_path"], header=None).to_numpy().squeeze()
        depth_timestamps = convert_timestamps_to_seconds(depth_timestamps, metadata=metadata)

        # Calculate sparse timestamps from linear alignment
        DOWN_FS = metadata["Constants"]["DEMODULATED_PHOTOMETRY_SAMPLING_RATE"]
        raw_fs = metadata["FiberPhotometry"]["raw_rate"]
        raw_indices = []
        for i, _ in enumerate(depth_timestamps):
            tdt_down_index = i * metadata["Alignment"]["slope"] + metadata["Alignment"]["bias"]
            tdt_raw_index = tdt_down_index / DOWN_FS * raw_fs
            raw_indices.append(int(tdt_raw_index))
        raw_indices = np.array(raw_indices)
        raw_indices = raw_indices[raw_indices < len(timestamps)]
        sparse_timestamps = depth_timestamps[np.arange(len(raw_indices))]
        start_time = metadata["Alignment"]["bias"] / metadata["Constants"]["DEMODULATED_PHOTOMETRY_SAMPLING_RATE"]
        sparse_timestamps += start_time
        sparse_timestamps = np.concatenate((np.array([0]), sparse_timestamps))
        raw_indices = np.concatenate((np.array([0]), raw_indices))

        # Interpolate aligned timestamps to raw photometry data
        temporal_interpolator = interp1d(raw_indices, sparse_timestamps, kind="linear", fill_value="extrapolate")
        aligned_timestamps = temporal_interpolator(np.arange(len(timestamps)))
        self.set_aligned_timestamps(aligned_timestamps=aligned_timestamps)

        return self.aligned_timestamps

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict) -> None:
        # Fibers Table
        fibers_table = FibersTable(
            description=(
                "Fiber photometry data with 2 excitation sources (470nm and 405nm), 1 PMT photodetector with "
                "a peak wavelength of 527nm, and 1 fluorophore (dLight1.1)."
            )
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
        if "FiberPhotometry" in metadata and "gain" in metadata["FiberPhotometry"]:
            photodetectors_table.add_row(peak_wavelength=527.0, type="PMT", gain=metadata["FiberPhotometry"]["gain"])
        else:
            photodetectors_table.add_row(peak_wavelength=527.0, type="PMT")

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

        keypoint_sessions = [
            "keypoint-dls-dlight-9",
            "keypoint-dls-dlight-10",
            "keypoint-dls-dlight-11",
            "keypoint-dls-dlight-12",
            "keypoint-dls-dlight-13",
        ]
        skip = (
            self.source_data["tdt_path"] is None
            or not Path(self.source_data["tdt_path"]).exists()
            or (
                not (Path(self.source_data["tdt_path"]).parent / "alignment_df.parquet").exists()
                and self.source_data["session_id"] not in keypoint_sessions
            )
        )
        if skip:
            excitation_sources_table.add_row(peak_wavelength=470.0, source_type="LED")
            excitation_sources_table.add_row(peak_wavelength=405.0, source_type="LED")

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
            )
            fibers_table.add_fiber(
                excitation_source=1,  # integers indicated rows of excitation sources table
                photodetector=0,
                fluorophores=[0],  # potentially multiple fluorophores, so list of indices
                location=metadata["FiberPhotometry"]["area"],
            )
            self.fibers_ref = DynamicTableRegion(
                name="rois", data=[0, 1], description="source fibers", table=fibers_table
            )
            return

        photometry_dict = load_tdt_data(self.source_data["tdt_path"], fs=metadata["FiberPhotometry"]["raw_rate"])
        timestamps = self.align_raw_timestamps(metadata=metadata)
        ascending_timestamps_indices = np.argsort(timestamps)

        raw_photometry = photometry_dict["pmt00"]
        commanded_signal = photometry_dict["pmt00_x"]
        commanded_reference = photometry_dict["pmt01_x"]

        # Commanded Voltage
        multi_commanded_voltage = MultiCommandedVoltage()
        commanded_signal_series = multi_commanded_voltage.create_commanded_voltage_series(
            name="commanded_signal",
            description=(
                "A 470nm (blue) LED and a 405nM (UV) LED (Mightex) were sinusoidally modulated at 161Hz and 381Hz, "
                "respectively (these frequencies were chosen to avoid harmonic cross-talk)."
            ),
            data=H5DataIO(commanded_signal[ascending_timestamps_indices], compression=True),
            frequency=metadata["FiberPhotometry"]["signal_freq"],
            power=float(metadata["FiberPhotometry"]["signal_amp"]),  # TODO: Fix this in ndx-photometry
            timestamps=H5DataIO(timestamps[ascending_timestamps_indices], compression=True),
            unit="volts",
        )
        commanded_reference_series = multi_commanded_voltage.create_commanded_voltage_series(
            name="commanded_reference",
            description=(
                "A 470nm (blue) LED and a 405nM (UV) LED (Mightex) were sinusoidally modulated at 161Hz and 381Hz, "
                "respectively (these frequencies were chosen to avoid harmonic cross-talk)."
            ),
            data=H5DataIO(commanded_reference[ascending_timestamps_indices], compression=True),
            frequency=metadata["FiberPhotometry"]["reference_freq"],
            power=float(metadata["FiberPhotometry"]["reference_amp"]),  # TODO: Fix this in ndx-photometry
            timestamps=commanded_signal_series.timestamps,
            unit="volts",
        )

        excitation_sources_table.add_row(
            peak_wavelength=470.0,
            source_type="LED",
            commanded_voltage=commanded_signal_series,
        )
        excitation_sources_table.add_row(
            peak_wavelength=405.0,
            source_type="LED",
            commanded_voltage=commanded_reference_series,
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
        )
        fibers_table.add_fiber(
            excitation_source=1,  # integers indicated rows of excitation sources table
            photodetector=0,
            fluorophores=[0],  # potentially multiple fluorophores, so list of indices
            location=metadata["FiberPhotometry"]["area"],
        )

        self.fibers_ref = DynamicTableRegion(name="rois", data=[0, 1], description="source fibers", table=fibers_table)

        # ROI Response Series
        # Here we set up a list of fibers that our recording came from
        raw_photometry = RoiResponseSeries(
            name="RawPhotometry",
            description="The raw acquisition with mixed signal from both the blue light excitation (470nm) and UV excitation (405nm).",
            comments=("Note: Raw photometry data is not temporally aligned for keypoint sessions."),
            data=H5DataIO(raw_photometry[ascending_timestamps_indices], compression=True),
            unit="F",
            timestamps=commanded_signal_series.timestamps,
            rois=self.fibers_ref,
        )

        # Aggregate into OPhys Module and NWBFile
        nwbfile.add_acquisition(raw_photometry)
        ophys_module = nwb_helpers.get_module(nwbfile, name="ophys", description="Fiber photometry data")
        ophys_module.add(multi_commanded_voltage)


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
