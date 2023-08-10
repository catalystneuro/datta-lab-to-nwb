"""Primary NWBConverter class for this dataset."""
from neuroconv import NWBConverter
from .fiberphotometryinterface import FiberPhotometryInterface
from .optogeneticinterface import OptogeneticInterface
from .velocitymodulationinterface import VelocityModulationInterface
from .behavioralsyllableinterface import BehavioralSyllableInterface
from .moseqextractinterface import MoseqExtractInterface
from .depthvideointerface import DepthVideoInterface
from .irvideointerface import IRVideoInterface


class NWBConverter(NWBConverter):
    """Primary conversion class."""

    data_interface_classes = dict(
        BehavioralSyllable=BehavioralSyllableInterface,
        FiberPhotometry=FiberPhotometryInterface,
        Optogenetic=OptogeneticInterface,
        VelocityModulation=VelocityModulationInterface,
        MoseqExtract=MoseqExtractInterface,
        DepthVideo=DepthVideoInterface,
        IRVideo=IRVideoInterface,
    )
