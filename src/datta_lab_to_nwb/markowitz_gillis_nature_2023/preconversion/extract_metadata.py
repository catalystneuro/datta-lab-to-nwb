# Scientific libraries
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import pytz
from neuroconv.utils import load_dict_from_file


def extract_photometry_metadata(data_path: str, example_uuid: str = None, num_sessions: int = None) -> dict:
    photometry_data_path = Path(data_path) / "dlight_raw_data/dlight_photometry_processed_full.parquet"
    columns = (
        "uuid",
        "session_name",
        "SessionName",
        "date",
        "mouse_id",
        "signal_max",
        "reference_max",
        "signal_reference_corr",
        "snr",
        "area",
    )
    metadata_columns = (
        "date",
        "mouse_id",
        "signal_max",
        "reference_max",
        "signal_reference_corr",
        "snr",
        "area",
    )
    photometry_reinforcement_mouse_ids = {
        "dlight-chrimson-1",
        "dlight-chrimson-2",
        "dlight-chrimson-3",
        "dlight-chrimson-4",
        "dlight-chrimson-5",
        "dlight-chrimson-6",
        "dlight-chrimson-8",
        "dlight-chrimson-9",
    }
    if example_uuid is None:
        uuid_df = pd.read_parquet(
            photometry_data_path,
            columns=["uuid", "mouse_id"],
            filters=[("mouse_id", "not in", photometry_reinforcement_mouse_ids)],
        )
        uuids = set(uuid_df.uuid[uuid_df.uuid.notnull()])
        del uuid_df
    else:
        uuids = {example_uuid}
    metadata = {}
    i = 0
    for uuid in tqdm(uuids):
        i += 1
        extract_session_metadata(columns, photometry_data_path, metadata, metadata_columns, uuid)
        metadata[uuid]["photometry_area"] = metadata[uuid].pop("area")
        if i >= num_sessions:
            break

    return metadata


def extract_session_metadata(columns, data_path, metadata, metadata_columns, uuid):
    """Extract metadata from a single session specified by uuid.

    Parameters
    ----------
    columns : tuple
        Columns to load from the parquet file.
    data_path : str
        Path to the parquet file.
    metadata : dict
        Dictionary to store the metadata in (edited in-place).
    metadata_columns : tuple
        Columns to extract metadata from.
    uuid : str
        UUID of the session to extract metadata from.
    """
    timezone = pytz.timezone("America/New_York")
    session_df = pd.read_parquet(
        data_path,
        columns=columns,
        filters=[("uuid", "==", uuid)],
    )
    metadata[uuid] = {}
    for col in metadata_columns:
        try:
            first_notnull = session_df.loc[session_df[col].notnull(), col].iloc[0]
        except IndexError:  # No non-null values found
            first_notnull = np.NaN
        if isinstance(first_notnull, np.float64):  # numpy scalars aren't serializable
            first_notnull = first_notnull.item()
        metadata[uuid][col] = first_notnull
    session_name = set(session_df.session_name[session_df.session_name.notnull()]) | set(
        session_df.SessionName[session_df.SessionName.notnull()]
    )
    assert len(session_name) <= 1, "Multiple session names found"
    try:
        metadata[uuid]["session_description"] = session_name.pop()
    except KeyError:  # No session name found
        metadata[uuid]["session_description"] = ""
    date = timezone.localize(metadata[uuid].pop("date"))
    metadata[uuid]["session_start_time"] = date.isoformat()
    metadata[uuid]["subject_id"] = metadata[uuid].pop("mouse_id")


if __name__ == "__main__":
    data_path = Path("/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior")
    photometry_metadata_path = Path(
        "/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior/metadata/photometry_metadata.yaml"
    )
    example_uuid = "2891f649-4fbd-4119-a807-b8ef507edfab"
    metadata = extract_photometry_metadata(data_path, num_sessions=10)
    with open(photometry_metadata_path, "w") as f:
        yaml.dump(metadata, f)

    # Test
    metadata = load_dict_from_file(photometry_metadata_path)
