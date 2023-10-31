"""Convenient utility functions for datta lab conversion common to multiple interfaces."""

import pandas as pd
import numpy as np


def convert_timestamps_to_seconds(timestamps: np.ndarray[int], metadata: dict) -> np.ndarray:
    """Converts integer timestamps to seconds using the metadata file.

    Parameters
    ----------
    timestamps : np.ndarray[int]
        The timestamps to convert.
    metadata : dict
        The metadata file.

    Returns
    -------
    np.ndarray
        The converted timestamps.
    """
    TIMESTAMPS_TO_SECONDS = metadata["Constants"]["TIMESTAMPS_TO_SECONDS"]
    timestamps[timestamps < timestamps[0]] = (
        metadata["Constants"]["MAXIMUM_TIMESTAMP"] + timestamps[timestamps < timestamps[0]]
    )
    timestamps -= timestamps[0]
    timestamps = timestamps * TIMESTAMPS_TO_SECONDS
    return timestamps
