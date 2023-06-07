"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter

from datta_lab_to_nwb.markowitz_gillis_nature_2023 import MarkowitzGillisNature2023BehaviorInterface
from datta_lab_to_nwb.markowitz_gillis_nature_2023 import FiberPhotometryInterface


class MarkowitzGillisNature2023NWBConverter(NWBConverter):
    """Primary conversion class."""

    data_interface_classes = dict(
        Behavior=MarkowitzGillisNature2023BehaviorInterface,
        FiberPhotometry=FiberPhotometryInterface,
    )
