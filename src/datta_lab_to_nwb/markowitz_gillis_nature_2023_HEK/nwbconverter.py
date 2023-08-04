"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter
from .hekinterface import HEKInterface


class NWBConverter(NWBConverter):
    """Primary conversion class."""

    data_interface_classes = dict(
        HEK=HEKInterface,
    )
