# Scientific libraries
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import pytz
from neuroconv.utils import load_dict_from_file
from neuroconv.utils import dict_deep_update


def extract_photometry_metadata(
    data_path: str, example_uuid: str = None, num_sessions: int = None, reinforcement_photometry: bool = False
) -> dict:
    """Extract metadata from photometry data.

    Parameters
    ----------
    data_path : str
        Path to data.
    example_uuid : str, optional
        UUID of example session to extract metadata from.
    num_sessions : int, optional
        Number of sessions to extract metadata from.
    reinforcement_photometry : bool, optional
        If True, extract metadata from reinforcement photometry sessions. If False, extract metadata from
        non-reinforcement photometry sessions.

    Returns
    -------
    metadata : dict
        Dictionary of metadata.
    """
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
    if reinforcement_photometry:
        filters = [("mouse_id", "in", photometry_reinforcement_mouse_ids)]
    else:
        filters = [("mouse_id", "not in", photometry_reinforcement_mouse_ids)]
    if example_uuid is None:
        uuid_df = pd.read_parquet(
            photometry_data_path,
            columns=["uuid", "mouse_id"],
            filters=filters,
        )
        uuids = set(uuid_df.uuid[uuid_df.uuid.notnull()])
        del uuid_df
    else:
        uuids = {example_uuid}
    if num_sessions is None:
        num_sessions = len(uuids)
    metadata = {}
    for i, uuid in enumerate(tqdm(uuids)):
        extract_session_metadata(columns, photometry_data_path, metadata, metadata_columns, uuid)
        metadata[uuid]["photometry_area"] = metadata[uuid].pop("area")
        if i >= num_sessions:
            break

    return metadata


def extract_reinforcement_metadata(
    data_path: str, example_uuid: str = None, num_sessions: int = None, reinforcement_photometry: bool = False
) -> dict:
    """Extract metadata from reinforcement data.

    Parameters
    ----------
    data_path : str
        Path to data.
    example_uuid : str, optional
        UUID of example session to extract metadata from.
    num_sessions : int, optional
        Number of sessions to extract metadata from.
    reinforcement_photometry : bool, optional
        If True, extract metadata from reinforcement photometry sessions. If False, extract metadata from
        non-photometry reinforcement sessions.

    Returns
    -------
    metadata : dict
        Dictionary of metadata.
    """
    reinforcement_data_path = Path(data_path) / "optoda_raw_data/closed_loop_behavior.parquet"
    columns = (
        "uuid",
        "SessionName",
        "date",
        "opsin",
        "genotype",
        "area",
        "mouse_id",
        "sex",
        "exclude",
        "cohort",
        "stim_duration",
        "stim_frequency",
        "pulse_width",
        "power",
    )
    metadata_columns = (
        "SessionName",
        "date",
        "opsin",
        "genotype",
        "area",
        "mouse_id",
        "sex",
        "exclude",
        "cohort",
        "stim_duration",
        "stim_frequency",
        "pulse_width",
        "power",
    )
    if reinforcement_photometry:
        filters = [("experiment_type", "==", "reinforcement_photometry")]
    else:
        filters = [("experiment_type", "==", "reinforcement")]
    if example_uuid is None:
        uuid_df = pd.read_parquet(
            reinforcement_data_path,
            columns=["uuid", "experiment_type"],
            filters=filters,
        )
        uuids = set(uuid_df.uuid[uuid_df.uuid.notnull()])
        del uuid_df
    else:
        uuids = {example_uuid}
    metadata = {}
    if num_sessions is None:
        num_sessions = len(uuids)
    for i, uuid in enumerate(tqdm(uuids)):
        extract_session_metadata(columns, reinforcement_data_path, metadata, metadata_columns, uuid)
        metadata[uuid]["optogenetic_area"] = metadata[uuid].pop("area")
        if i >= num_sessions:
            break

    return metadata


def extract_reinforcement_photometry_metadata(
    data_path: str, example_uuid: str = None, num_sessions: int = None
) -> dict:
    """Extract metadata from reinforcement photometry data.

    Parameters
    ----------
    data_path : str
        Path to data.
    example_uuid : str, optional
        UUID of example session to extract metadata from.
    num_sessions : int, optional
        Number of sessions to extract metadata from.

    Returns
    -------
    metadata : dict
        Dictionary of metadata.
    """
    photometry_metadata = extract_photometry_metadata(
        data_path, example_uuid, num_sessions, reinforcement_photometry=True
    )
    reinforcement_metadata = extract_reinforcement_metadata(
        data_path, example_uuid, num_sessions, reinforcement_photometry=True
    )
    photometry_uuids = set(photometry_metadata.keys())
    reinforcement_uuids = set(reinforcement_metadata.keys())
    metadata = {}
    for uuid in photometry_uuids:
        for photometry_key in photometry_metadata[uuid].keys():
            try:
                assert (
                    photometry_metadata[uuid][photometry_key] == reinforcement_metadata[uuid][photometry_key]
                ), f"photometry metadata and reinforcement metadata don't match (photometry[{uuid}][{photometry_key}]: {photometry_metadata[uuid][photometry_key]}, reinforcement[{uuid}][{photometry_key}]: {reinforcement_metadata[uuid][photometry_key]})"
            except KeyError:  # reinforcement metadata doesn't have this uuid and/or metadata field
                pass
            try:
                metadata[uuid][photometry_key] = photometry_metadata[uuid][photometry_key]
            except KeyError:  # New uuid
                metadata[uuid] = {}
                metadata[uuid][photometry_key] = photometry_metadata[uuid][photometry_key]
    for uuid in reinforcement_uuids:
        for reinforcement_key in reinforcement_metadata[uuid].keys():
            try:
                assert (
                    photometry_metadata[uuid][reinforcement_key] == reinforcement_metadata[uuid][reinforcement_key]
                ), f"photometry metadata and reinforcement metadata don't match (photometry[{uuid}][{reinforcement_key}]: {photometry_metadata[uuid][reinforcement_key]}, reinforcement[{uuid}][{reinforcement_key}]: {reinforcement_metadata[uuid][reinforcement_key]})"
            except KeyError:
                pass
            try:
                metadata[uuid][reinforcement_key] = reinforcement_metadata[uuid][reinforcement_key]
            except KeyError:
                metadata[uuid] = {}
                metadata[uuid][reinforcement_key] = reinforcement_metadata[uuid][reinforcement_key]
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
        try:  # numpy arrays aren't serializable --> extract relevant value with .item()
            first_notnull = first_notnull.item()
        except AttributeError:
            pass
        metadata[uuid][col] = first_notnull
    if "session_name" in columns and "SessionName" in columns:  # session name is duplicated (photometry data)
        metadata[uuid]["session_description"] = get_session_name(session_df)
    else:
        metadata[uuid]["session_description"] = metadata[uuid].pop("SessionName")
    date = timezone.localize(metadata[uuid].pop("date"))
    metadata[uuid]["session_start_time"] = date.isoformat()
    metadata[uuid]["subject_id"] = metadata[uuid].pop("mouse_id")


def get_session_name(session_df):
    """Get the session name from a dataframe containing both "session_name" and "SessionName" columns.

    Parameters
    ----------
    session_df : pandas.DataFrame
        Dataframe containing both "session_name" and "SessionName" columns.

    Returns
    -------
    session_name : str
        Session name.
    """
    session_names = set(session_df.session_name[session_df.session_name.notnull()]) | set(
        session_df.SessionName[session_df.SessionName.notnull()]
    )
    assert len(session_names) <= 1, "Multiple session names found"
    try:
        session_name = session_names.pop()
    except KeyError:  # No session name found
        session_name = ""
    return session_name


if __name__ == "__main__":
    data_path = Path("/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior")
    photometry_metadata_path = Path(
        "/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior/metadata/photometry_metadata.yaml"
    )
    reinforcement_metadata_path = Path(
        "/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior/metadata/reinforcement_metadata.yaml"
    )
    reinforcement_photometry_metadata_path = Path(
        "/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior/metadata/reinforcement_photometry_metadata.yaml"
    )
    example_uuid = "2891f649-4fbd-4119-a807-b8ef507edfab"
    reinforcement_metadata = extract_reinforcement_metadata(data_path, num_sessions=3)
    photometry_metadata = extract_photometry_metadata(data_path, num_sessions=3)
    reinforcement_photometry_metadata = extract_reinforcement_photometry_metadata(data_path, example_uuid=example_uuid)
    with open(photometry_metadata_path, "w") as f:
        yaml.dump(photometry_metadata, f)
    with open(reinforcement_metadata_path, "w") as f:
        yaml.dump(reinforcement_metadata, f)
    with open(reinforcement_photometry_metadata_path, "w") as f:
        yaml.dump(reinforcement_photometry_metadata, f)
