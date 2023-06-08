"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter
from datta_lab_to_nwb import markowitz_gillis_nature_2023


class NWBConverter(NWBConverter):
    """Primary conversion class."""

    data_interface_classes = dict(
        Behavior=markowitz_gillis_nature_2023.BehaviorInterface,
        FiberPhotometry=markowitz_gillis_nature_2023.FiberPhotometryInterface,
    )
