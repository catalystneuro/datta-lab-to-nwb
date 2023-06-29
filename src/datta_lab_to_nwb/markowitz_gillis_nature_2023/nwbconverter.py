"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter
from .behaviorinterface import BehaviorInterface
from .fiberphotometryinterface import FiberPhotometryInterface
from .optogeneticinterface import OptogeneticInterface
from .behavioralsyllableinterface import BehavioralSyllableInterface
from .metadatainterface import MetadataInterface


class NWBConverter(NWBConverter):
    """Primary conversion class."""

    data_interface_classes = dict(
        Metadata=MetadataInterface,
        Behavior=BehaviorInterface,
        BehavioralSyllable=BehavioralSyllableInterface,
        FiberPhotometry=FiberPhotometryInterface,
        Optogenetic=OptogeneticInterface,
    )
