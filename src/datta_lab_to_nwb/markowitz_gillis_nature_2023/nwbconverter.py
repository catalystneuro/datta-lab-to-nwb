"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter
from .behaviorinterface import BehaviorInterface
from .fiberphotometryinterface import FiberPhotometryInterface


class NWBConverter(NWBConverter):
    """Primary conversion class."""

    data_interface_classes = dict(
        Behavior=BehaviorInterface,
        FiberPhotometry=FiberPhotometryInterface,
    )
