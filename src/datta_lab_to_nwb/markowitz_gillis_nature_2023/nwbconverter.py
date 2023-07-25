"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter
from .behaviorinterface import BehaviorInterface
from .fiberphotometryinterface import FiberPhotometryInterface
from .optogeneticinterface import OptogeneticInterface
from .behavioralsyllableinterface import BehavioralSyllableInterface
from .moseqinterface import MoseqInterface


class NWBConverter(NWBConverter):
    """Primary conversion class."""

    data_interface_classes = dict(
        Behavior=BehaviorInterface,
        BehavioralSyllable=BehavioralSyllableInterface,
        FiberPhotometry=FiberPhotometryInterface,
        Optogenetic=OptogeneticInterface,
        Moseq=MoseqInterface,
    )
