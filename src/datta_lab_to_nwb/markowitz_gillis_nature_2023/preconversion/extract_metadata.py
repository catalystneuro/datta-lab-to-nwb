# Scientific libraries
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import pytz
from neuroconv.utils import load_dict_from_file


def extract_photometry_metadata(
    data_path: str, example_uuid: str = None, num_sessions: int = None, exclude_reinforcement_photometry: bool = True
) -> dict:
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
    if exclude_reinforcement_photometry:
        filters = [("mouse_id", "not in", photometry_reinforcement_mouse_ids)]
    else:
        filters = None
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
    metadata = {}
    i = 0
    for uuid in tqdm(uuids):
        i += 1
        extract_session_metadata(columns, photometry_data_path, metadata, metadata_columns, uuid)
        metadata[uuid]["photometry_area"] = metadata[uuid].pop("area")
        if i >= num_sessions:
            break

    return metadata


def extract_reinforcement_metadata(
    data_path: str, example_uuid: str = None, num_sessions: int = None, exclude_reinforcement_photometry: bool = True
) -> dict:
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
    if exclude_reinforcement_photometry:
        filters = [("experiment_type", "==", "reinforcement")]
    else:
        filters = None
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
    i = 0
    for uuid in tqdm(uuids):
        i += 1
        extract_session_metadata(columns, reinforcement_data_path, metadata, metadata_columns, uuid)
        metadata[uuid]["optogenetic_area"] = metadata[uuid].pop("area")
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
    example_uuid = "2891f649-4fbd-4119-a807-b8ef507edfab"
    reinforcement_metadata = extract_reinforcement_metadata(data_path, num_sessions=3)
    photometry_metadata = extract_photometry_metadata(data_path, num_sessions=3)
    with open(photometry_metadata_path, "w") as f:
        yaml.dump(photometry_metadata, f)
    with open(reinforcement_metadata_path, "w") as f:
        yaml.dump(reinforcement_metadata, f)
