"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter
from ..markowitz_gillis_nature_2023.fiberphotometryinterface import RawFiberPhotometryInterface
from .irvideointerface import IRVideoInterface


class NWBConverter(NWBConverter):
    """Primary conversion class."""

    data_interface_classes = dict(
        FiberPhotometry=RawFiberPhotometryInterface,
        IRVideo=IRVideoInterface,
    )
