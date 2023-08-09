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
    data_path: str,
    example_uuids: str = None,
    num_sessions: int = None,
    reinforcement_photometry: bool = False,
) -> dict:
    """Extract metadata from photometry data.

    Parameters
    ----------
    data_path : str
        Path to data.
    example_uuids : str, optional
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
    session_columns = (
        "uuid",
        "session_name",
        "SessionName",
        "date",
        "mouse_id",
        "signal_max",
        "reference_max",
        "signal_reference_corr",
        "snr",
    )
    subject_columns = (
        "mouse_id",
        "genotype",
        "area",
        "opsin",
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
    if example_uuids is None:
        df = pd.read_parquet(
            photometry_data_path,
            columns=["uuid", "mouse_id"],
            filters=filters,
        )
        uuids = set(df.uuid[df.uuid.notnull()])
        del df
    else:
        uuids = set(example_uuids)
    if num_sessions is None:
        num_sessions = len(uuids)
    session_metadata = {}
    for i, uuid in enumerate(tqdm(uuids, desc="Extracting photometry session metadata")):
        extract_session_metadata(session_columns, photometry_data_path, session_metadata, uuid)
        session_metadata[uuid]["photometry"] = True
        if i + 1 >= num_sessions:
            break
    subject_ids = set(session_metadata[uuid]["subject_id"] for uuid in session_metadata)
    subject_metadata = {}
    for mouse_id in tqdm(subject_ids, desc="Extracting photometry subject metadata"):
        extract_subject_metadata(subject_columns, photometry_data_path, subject_metadata, mouse_id)
        subject_metadata[mouse_id]["photometry_area"] = subject_metadata[mouse_id].pop("area")
        subject_metadata[mouse_id]["sex"] = "U"

    return session_metadata, subject_metadata


def extract_reinforcement_metadata(
    data_path: str, example_uuids: str = None, num_sessions: int = None, reinforcement_photometry: bool = False
) -> dict:
    """Extract metadata from reinforcement data.

    Parameters
    ----------
    data_path : str
        Path to data.
    example_uuids : str, optional
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
    session_columns = (
        "uuid",
        "SessionName",
        "date",
        "mouse_id",
        "exclude",
        "stim_duration",
        "stim_frequency",
        "pulse_width",
        "power",
    )
    subject_columns = (
        "mouse_id",
        "sex",
        "opsin",
        "genotype",
        "area",
        "cohort",
    )
    if reinforcement_photometry:
        filters = [
            (
                "experiment_type",
                "in",
                {"reinforcement_photometry", "excitation_photometry", "excitation_pulsed_photometry"},
            )
        ]
    else:
        filters = [("experiment_type", "in", {"reinforcement", "excitation", "excitation_pulsed"})]
    if example_uuids is None:
        df = pd.read_parquet(
            reinforcement_data_path,
            columns=["uuid", "experiment_type", "mouse_id"],
            filters=filters,
        )
        uuids = set(df.uuid[df.uuid.notnull()])
        del df
    else:
        uuids = set(example_uuids)
    session_metadata, subject_metadata = {}, {}
    if num_sessions is None:
        num_sessions = len(uuids)
    for i, uuid in enumerate(tqdm(uuids, desc="Extracting reinforcement session metadata")):
        extract_session_metadata(session_columns, reinforcement_data_path, session_metadata, uuid)
        # add si units to names
        session_metadata[uuid]["stim_duration_s"] = session_metadata[uuid].pop("stim_duration")
        session_metadata[uuid]["stim_frequency_Hz"] = session_metadata[uuid].pop("stim_frequency")
        session_metadata[uuid]["pulse_width_s"] = session_metadata[uuid].pop("pulse_width")
        session_metadata[uuid]["power_watts"] = session_metadata[uuid].pop("power") / 1000
        session_metadata[uuid]["reinforcement"] = True
        if i + 1 >= num_sessions:
            break
    subject_ids = set(session_metadata[uuid]["subject_id"] for uuid in session_metadata)
    for mouse_id in tqdm(subject_ids, desc="Extracting reinforcement subject metadata"):
        extract_subject_metadata(subject_columns, reinforcement_data_path, subject_metadata, mouse_id)
        subject_metadata[mouse_id]["optogenetic_area"] = subject_metadata[mouse_id].pop("area")
        if subject_metadata[mouse_id]["sex"] == "male":
            subject_metadata[mouse_id]["sex"] = "M"
        elif subject_metadata[mouse_id]["sex"] == "female":
            subject_metadata[mouse_id]["sex"] = "F"
        else:
            subject_metadata[mouse_id]["sex"] = "U"

    return session_metadata, subject_metadata


def extract_velocity_modulation_metadata(
    data_path: str,
    example_uuids: str = None,
    num_sessions: int = None,
) -> dict:
    """Extract metadata from velocity modulation data.
    Parameters
    ----------
    data_path : str
        Path to data.
    example_uuids : str, optional
        UUID of example session to extract metadata from.
    num_sessions : int, optional
        Number of sessions to extract metadata from.
    Returns
    -------
    metadata : dict
        Dictionary of metadata.
    """
    velocity_data_path = Path(data_path) / "optoda_raw_data/closed_loop_behavior_velocity_conditioned.parquet"
    session_columns = (
        "uuid",
        "SessionName",
        "date",
        "mouse_id",
        "stim_duration",
        "trigger_syllable",
        "trigger_syllable_scalar_threshold",
        "trigger_syllable_scalar_comparison",
    )
    subject_columns = (
        "mouse_id",
        "genotype",
        "cohort",
    )
    if example_uuids is None:
        df = pd.read_parquet(
            velocity_data_path,
            columns=["uuid"],
        )
        uuids = set(df.uuid[df.uuid.notnull()])
        del df
    else:
        uuids = set(example_uuids)
    session_metadata, subject_metadata = {}, {}
    if num_sessions is None:
        num_sessions = len(uuids)
    for i, uuid in enumerate(tqdm(uuids, desc="Extracting velocity-modulation session metadata")):
        extract_session_metadata(session_columns, velocity_data_path, session_metadata, uuid)
        # add si units to names
        session_metadata[uuid]["stim_duration_s"] = session_metadata[uuid].pop("stim_duration")
        session_metadata[uuid]["stim_frequency_Hz"] = np.NaN
        session_metadata[uuid]["pulse_width_s"] = np.NaN
        session_metadata[uuid]["power_watts"] = 10 / 1000  # power = 10mW from paper
        session_metadata[uuid]["reinforcement"] = True
        if i + 1 >= num_sessions:
            break
    subject_ids = set(session_metadata[uuid]["subject_id"] for uuid in session_metadata)
    for mouse_id in tqdm(subject_ids, desc="Extracting reinforcement subject metadata"):
        extract_subject_metadata(subject_columns, velocity_data_path, subject_metadata, mouse_id)
        subject_metadata[mouse_id]["optogenetic_area"] = "snc (axon)"  # from paper
        subject_metadata[mouse_id]["sex"] = "U"

    return session_metadata, subject_metadata


def extract_reinforcement_photometry_metadata(
    data_path: str, example_uuids: str = None, num_sessions: int = None
) -> dict:
    """Extract metadata from reinforcement photometry data.

    Parameters
    ----------
    data_path : str
        Path to data.
    example_uuids : str, optional
        UUID of example session to extract metadata from.
    num_sessions : int, optional
        Number of sessions to extract metadata from.

    Returns
    -------
    metadata : dict
        Dictionary of metadata.
    """
    photometry_session_metadata, photometry_subject_metadata = extract_photometry_metadata(
        data_path, example_uuids, num_sessions, reinforcement_photometry=True
    )
    reinforcement_session_metadata, reinforcement_subject_metadata = extract_reinforcement_metadata(
        data_path, example_uuids, num_sessions, reinforcement_photometry=True
    )
    photometry_uuids = set(photometry_session_metadata.keys())
    reinforcement_uuids = set(reinforcement_session_metadata.keys())
    session_metadata = resolve_duplicates(
        photometry_session_metadata, photometry_uuids, reinforcement_session_metadata, reinforcement_uuids
    )
    photometry_subject_ids = set(photometry_subject_metadata.keys())
    reinforcement_subject_ids = set(reinforcement_subject_metadata.keys())
    subject_metadata = resolve_duplicates(
        photometry_subject_metadata, photometry_subject_ids, reinforcement_subject_metadata, reinforcement_subject_ids
    )
    return session_metadata, subject_metadata


def resolve_duplicates(photometry_metadata, photometry_ids, reinforcement_metadata, reinforcement_ids):
    resolved_metadata = {}
    _resolve_duplicates(
        resolved_metadata, photometry_ids, photometry_metadata, reinforcement_ids, reinforcement_metadata
    )
    _resolve_duplicates(
        resolved_metadata, reinforcement_ids, reinforcement_metadata, photometry_ids, photometry_metadata
    )
    return resolved_metadata


def _resolve_duplicates(resolved_dict, ids1, dict1, ids2, dict2):
    for id1 in ids1:
        if id1 not in ids2:
            resolved_dict[id1] = dict1[id1]
            continue
        if id1 not in resolved_dict:
            resolved_dict[id1] = {}
        for key1 in dict1[id1].keys():
            if key1 in dict2[id1].keys():
                try:
                    assert (
                        dict1[id1][key1] == dict2[id1][key1]
                    ), f"dict1 and dict2 don't match (dict1[{id1}][{key1}]: {dict1[id1][key1]}, dict2[{id1}][{key1}]: {dict2[id1][key1]})"
                except AssertionError:
                    assert key1 == "sex"
                    if dict1[id1][key1] == "U":
                        dict1[id1][key1] = dict2[id1][key1]
            resolved_dict[id1][key1] = dict1[id1][key1]


def extract_session_metadata(columns, data_path, metadata, uuid):
    """Extract metadata from a single session specified by uuid.

    Parameters
    ----------
    columns : tuple
        Columns to load from the parquet file.
    data_path : str
        Path to the parquet file.
    metadata : dict
        Dictionary to store the metadata in (edited in-place).
    uuid : str
        UUID of the session to extract metadata from.
    """
    timezone = pytz.timezone("America/New_York")
    session_df = extract_metadata(columns, data_path, metadata, uuid, "uuid")
    metadata[uuid]["session_description"] = get_session_name(session_df)
    date = timezone.localize(metadata[uuid].pop("date"))
    metadata[uuid]["session_start_time"] = date.isoformat()
    metadata[uuid]["subject_id"] = metadata[uuid].pop("mouse_id")


def extract_subject_metadata(columns, data_path, metadata, subject_id):
    """Extract metadata from a single subject specified by subject_id.

    Parameters
    ----------
    columns : tuple
        Columns to load from the parquet file.
    data_path : str
        Path to the parquet file.
    metadata : dict
        Dictionary to store the metadata in (edited in-place).
    subject_id : str
        Subject ID of the subject to extract metadata from.
    """
    extract_metadata(columns, data_path, metadata, subject_id, "mouse_id")


def extract_metadata(columns, data_path, metadata, key, key_name):
    df = pd.read_parquet(
        data_path,
        columns=columns,
        filters=[(key_name, "==", key)],
    )
    metadata[key] = {}
    for col in columns:
        if col in {key_name, "session_name", "SessionName"}:
            continue
        any_notnull = len(df.loc[df[col].notnull(), col]) > 0
        if any_notnull:
            first_notnull = df.loc[df[col].notnull(), col].iloc[0]
        else:
            first_notnull = np.NaN
        if isinstance(first_notnull, np.generic):  # numpy scalars aren't serializable --> extract value with .item()
            first_notnull = first_notnull.item()
        metadata[key][col] = first_notnull
    return df


def get_session_name(session_df):
    """Get the session name from a dataframe potentially containing both "session_name" and "SessionName" columns.

    Parameters
    ----------
    session_df : pandas.DataFrame
        Dataframe containing both "session_name" and "SessionName" columns.

    Returns
    -------
    session_name : str
        Session name.
    """
    session_names = set(session_df.SessionName[session_df.SessionName.notnull()])  # SessionName is always present
    if "session_name" in session_df.columns:
        session_names = session_names.union(set(session_df.session_name[session_df.session_name.notnull()]))
    assert len(session_names) <= 1, "Multiple session names found"
    any_session_names = len(session_names) > 0
    if any_session_names:
        session_name = session_names.pop()
    else:
        session_name = ""
    return session_name


if __name__ == "__main__":
    data_path = Path("/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior")
    metadata_path = data_path / "metadata"
    photometry_session_metadata_path = metadata_path / "photometry_session_metadata.yaml"
    photometry_subject_metadata_path = metadata_path / "photometry_subject_metadata.yaml"
    reinforcement_session_metadata_path = metadata_path / "reinforcement_session_metadata.yaml"
    reinforcement_subject_metadata_path = metadata_path / "reinforcement_subject_metadata.yaml"
    reinforcement_photometry_session_metadata_path = metadata_path / "reinforcement_photometry_session_metadata.yaml"
    reinforcement_photometry_subject_metadata_path = metadata_path / "reinforcement_photometry_subject_metadata.yaml"
    velocity_session_metadata_path = metadata_path / "velocity_modulation_session_metadata.yaml"
    velocity_subject_metadata_path = metadata_path / "velocity_modulation_subject_metadata.yaml"

    # Example UUIDs
    dls_dlight_1_example = "18dc5ad5-13f0-4297-8b21-75d434770e57"
    photometry_examples = [dls_dlight_1_example]
    reinforcement_example = "dcf0767a-b75d-4c79-a242-84dd5b5cdd00"
    excitation_example = "380d4711-85a6-4672-ad48-76e91607c41f"
    excitation_pulsed_example = "be01945e-c6d0-4bca-bd56-4d4466d9d832"
    reinforcement_examples = [reinforcement_example, excitation_example, excitation_pulsed_example]
    figure1d_example = "2891f649-4fbd-4119-a807-b8ef507edfab"
    pulsed_photometry_example = "b8360fcd-acfd-4414-9e67-ba0dc5c979a8"
    excitation_photometry_example = "95bec433-2242-4276-b8a5-6d069afa3910"
    raw_fp_example = "b814a426-7ec9-440e-baaa-105ba27a5fa6"
    reinforcement_photometry_examples = [
        figure1d_example,
        pulsed_photometry_example,
        excitation_photometry_example,
        raw_fp_example,
    ]
    velocity_modulation_examples = ["c621e134-50ec-4e8b-8175-a8c023d92789"]

    reinforcement_session_metadata, reinforcement_subject_metadata = extract_reinforcement_metadata(
        data_path,
        example_uuids=reinforcement_examples,
    )
    photometry_session_metadata, photometry_subject_metadata = extract_photometry_metadata(
        data_path,
        example_uuids=photometry_examples,
    )
    (
        reinforcement_photometry_session_metadata,
        reinforcement_photometry_subject_metadata,
    ) = extract_reinforcement_photometry_metadata(data_path, example_uuids=reinforcement_photometry_examples)
    velocity_session_metadata, velocity_subject_metadata = extract_velocity_modulation_metadata(
        data_path, example_uuids=velocity_modulation_examples
    )

    path2metadata = {
        photometry_session_metadata_path: photometry_session_metadata,
        photometry_subject_metadata_path: photometry_subject_metadata,
        reinforcement_session_metadata_path: reinforcement_session_metadata,
        reinforcement_subject_metadata_path: reinforcement_subject_metadata,
        reinforcement_photometry_session_metadata_path: reinforcement_photometry_session_metadata,
        reinforcement_photometry_subject_metadata_path: reinforcement_photometry_subject_metadata,
        velocity_session_metadata_path: velocity_session_metadata,
        velocity_subject_metadata_path: velocity_subject_metadata,
    }
    for path, resolved_dict in path2metadata.items():
        with open(path, "w") as f:
            yaml.dump(resolved_dict, f)
