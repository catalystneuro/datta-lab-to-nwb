"""Primary class for converting in-vitro photometry data."""

from typing import Literal, Optional
import numpy as np
from pynwb import NWBFile
from pynwb.ophys import (
    OnePhotonSeries,
    OpticalChannel,
)
from pynwb.image import GrayscaleImage, Images
from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools import nwb_helpers
from neuroconv.utils import load_dict_from_file
from datetime import datetime
import pytz
from pathlib import Path
from skimage.io import imread
from hdmf.backends.hdf5.h5_utils import H5DataIO


class HEKInterface(BaseDataInterface):
    """HEK interface for markowitz_gillis_nature_2023 conversion"""

    def __init__(self, file_path: str, scale_path: str):
        grid_spacing_um = 0.43  # Manually extracted from scale image (assuming x = y -- i.e. square grid)
        sampling_frequency = 0.25
        file_name_split = Path(file_path).name.split("_")
        date = file_name_split[2]
        experiment_name = file_name_split[1]
        date = datetime.strptime(date, "%Y%m%d")
        time = Path(file_path).name.split("_")[3]
        time = datetime.strptime(time, "%H%M%S")
        timezone = pytz.timezone("America/New_York")
        session_start_time = datetime.combine(date, time.time(), tzinfo=timezone)
        super().__init__(
            file_path=file_path,
            scale_path=scale_path,
            sampling_frequency=sampling_frequency,
            grid_spacing_um=grid_spacing_um,
            experiment_name=experiment_name,
            session_start_time=session_start_time,
        )

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()

        metadata["NWBFile"]["session_description"] = (
            "To test whether UV excitation could be a suitable reference wavelength for dLight1.1, HEK 293 cells "
            "(ATCC, cells were validated by ATCC via short tandem repeat analysis and were not tested for mycoplasma) "
            "were transfected with the dLight1.1 plasmid (Addgene 111067-AAV5) using Mirus TransIT-LT1 (MIR 2304). "
            "Cells were imaged using an Olympus BX51W I upright microscope and a LUMPlanFl/IR 60×/0.90W objective. "
            "Excitation light was delivered by an AURA light engine (Lumencor) at 400 and 480nm with 50ms exposure "
            "time. Emission light was split with an FF395/495/610-Di01 dichroic mirror and bandpass filtered with an "
            "FF01-425/527/685 filter (all filter optics from Semrock). Images were collected with a CCD camera "
            "(IMAGO-QE, Thermo Fisher Scientific), at a rate of one frame every two seconds, alternating the "
            "excitation wavelengths in each frame."
        )
        metadata["NWBFile"]["session_start_time"] = self.source_data["session_start_time"]
        metadata["NWBFile"]["identifier"] = self.source_data["experiment_name"]
        metadata["NWBFile"]["session_id"] = self.source_data["experiment_name"]

        return metadata

    def get_metadata_schema(self) -> dict:
        metadata_schema = super().get_metadata_schema()
        metadata_schema["properties"]["Behavior"] = {
            "type": "object",
            "properties": {
                "CompassDirection": {
                    "type": "object",
                    "properties": {
                        "reference_frame": {"type": "string"},
                    },
                },
                "Position": {
                    "type": "object",
                    "properties": {
                        "reference_frame": {"type": "string"},
                    },
                },
            },
        }
        metadata_schema["properties"]["BehavioralSyllable"] = {
            "type": "object",
            "properties": {
                "sorted_pseudoindex2name": {"type": "object"},
                "id2sorted_index": {"type": "object"},
                "sorted_index2id": {"type": "object"},
            },
        }
        return metadata_schema

    def add_to_nwbfile(self, nwbfile: NWBFile, metadata: dict) -> None:
        raw_data = imread(self.source_data["file_path"])
        raw_scale = imread(self.source_data["scale_path"])
        raw_signal = raw_data[1::2, :, :]
        raw_reference = raw_data[::2, :, :]

        device = nwbfile.create_device(
            name="Microscope",
            description=(
                "Cells were imaged using an Olympus BX51W I upright microscope and a LUMPlanFl/IR 60×/0.90W objective"
            ),
            manufacturer="Olympus",
        )
        signal_channel = OpticalChannel(
            name="SignalChannel",
            description=(
                "Excitation light was delivered by an AURA light engine (Lumencor) at 480nm with 50ms exposure time."
            ),
            emission_lambda=480.0,
        )
        reference_channel = OpticalChannel(
            name="ReferenceChannel",
            description=(
                "Excitation light was delivered by an AURA light engine (Lumencor) at 400nm with 50ms exposure time."
            ),
            emission_lambda=400.0,
        )
        signal_imaging_plane = nwbfile.create_imaging_plane(
            name="SignalImagingPlane",
            optical_channel=signal_channel,
            imaging_rate=self.source_data["sampling_frequency"],
            description=(
                "Emission light was split with an FF395/495/610-Di01 dichroic mirror and bandpass filtered with an "
                "FF01-425/527/685 filter (all filter optics from Semrock). Images were collected with a CCD camera "
                "(IMAGO-QE, Thermo Fisher Scientific), at a rate of one frame every two seconds, alternating the "
                "excitation wavelengths in each frame."
            ),
            device=device,
            excitation_lambda=480.0,
            indicator="dLight1.1",
            location="n.a.",
            grid_spacing=[self.source_data["grid_spacing_um"], self.source_data["grid_spacing_um"]],
            grid_spacing_unit="micrometers",
            origin_coords=[0.0, 0.0],
            origin_coords_unit="micrometers",
        )
        reference_imaging_plane = nwbfile.create_imaging_plane(
            name="ReferenceImagingPlane",
            optical_channel=reference_channel,
            imaging_rate=self.source_data["sampling_frequency"],
            description=(
                "Emission light was split with an FF395/495/610-Di01 dichroic mirror and bandpass filtered with an "
                "FF01-425/527/685 filter (all filter optics from Semrock). Images were collected with a CCD camera "
                "(IMAGO-QE, Thermo Fisher Scientific), at a rate of one frame every two seconds, alternating the "
                "excitation wavelengths in each frame."
            ),
            device=device,
            excitation_lambda=400.0,
            indicator="dLight1.1",
            location="n.a.",
            grid_spacing=[self.source_data["grid_spacing_um"], self.source_data["grid_spacing_um"]],
            grid_spacing_unit="micrometers",
            origin_coords=[0.0, 0.0],
            origin_coords_unit="micrometers",
        )
        MiB = 1024**2
        desired_nbytes = 10 * MiB
        chunk_size = desired_nbytes / raw_signal.itemsize
        chunks = (
            int(chunk_size / (raw_signal.shape[1] * raw_signal.shape[2])),
            raw_signal.shape[1],
            raw_signal.shape[2],
        )
        signal_1p_series = OnePhotonSeries(
            name="Signal1PSeries",
            data=H5DataIO(raw_signal, compression=True, chunks=chunks),
            imaging_plane=signal_imaging_plane,
            rate=self.source_data["sampling_frequency"],
            unit="normalized amplitude",
            description="Fluorescence signal corresponding to the 480nm excitation wavelength.",
        )
        chunk_size = desired_nbytes / raw_reference.itemsize
        chunks = (
            int(chunk_size / (raw_reference.shape[1] * raw_reference.shape[2])),
            raw_reference.shape[1],
            raw_reference.shape[2],
        )
        reference_1p_series = OnePhotonSeries(
            name="Reference1PSeries",
            data=H5DataIO(raw_reference, compression=True, chunks=chunks),
            imaging_plane=reference_imaging_plane,
            rate=self.source_data["sampling_frequency"],
            unit="normalized amplitude",
            description="Fluorescence isosbestic reference corresponding to the 400nm excitation wavelength.",
        )
        scale_image = GrayscaleImage(
            name="ScaleImage",
            data=H5DataIO(raw_scale, compression=True),
            description="Image of the scale bar: Lines represent 10um divisions.",
        )
        images = Images(name="Images", description="Container for scale image.", images=[scale_image])
        nwbfile.add_acquisition(signal_1p_series)
        nwbfile.add_acquisition(reference_1p_series)
        nwbfile.add_acquisition(images)
