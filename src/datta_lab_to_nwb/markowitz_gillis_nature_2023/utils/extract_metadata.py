from pathlib import Path
import pandas as pd
import yaml
from tqdm import tqdm
from neuroconv.utils import load_dict_from_file


def extract_metadata(data_path: str, metadata_path: str):
    columns = (
        "uuid",
        "session_name",
        "SessionName",
        "date",
        "mouse_id",
    )
    metadata_columns = (
        "date",
        "mouse_id",
    )
    uuid_df = pd.read_parquet(data_path, columns=["uuid"])
    uuids = set(uuid_df.uuid[uuid_df.uuid.notnull()])
    del uuid_df
    metadata = {}
    for uuid in tqdm(uuids):
        session_df = pd.read_parquet(
            data_path,
            columns=columns,
            filters=[("uuid", "==", uuid)],
        )
        metadata[uuid] = {}
        for col in metadata_columns:
            first_notnull = session_df.loc[session_df[col].notnull(), col].iloc[0]
            metadata[uuid][col] = first_notnull
        session_name = set(session_df.session_name[session_df.session_name.notnull()]) | set(
            session_df.SessionName[session_df.SessionName.notnull()]
        )
        assert len(session_name) <= 1, "Multiple session names found"
        try:
            metadata[uuid]["session_description"] = session_name.pop()
        except KeyError:  # No session name found
            metadata[uuid]["session_description"] = ""
        metadata[uuid]["session_start_time"] = metadata[uuid].pop("date").isoformat()
        metadata[uuid]["subject_id"] = metadata[uuid].pop("mouse_id")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)


if __name__ == "__main__":
    data_path = Path(
        "/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior/dlight_raw_data/dlight_photometry_processed_full.parquet"
    )
    metadata_path = Path(
        "/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior/dlight_raw_data/session_metadata.yaml"
    )
    extract_metadata(data_path, metadata_path)
    metadata = load_dict_from_file(metadata_path)
    print(metadata)
