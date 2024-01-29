"""Primary NWBConverter class for this dataset."""

from neuroconv import NWBConverter
from .fiberphotometryinterface import FiberPhotometryInterface
from .irvideointerface import IRVideoInterface
from .keypointinterface import KeypointInterface


class NWBConverter(NWBConverter):
    """Primary conversion class."""

    data_interface_classes = dict(
        FiberPhotometry=FiberPhotometryInterface,
        IRVideo=IRVideoInterface,
        Keypoint=KeypointInterface,
    )
