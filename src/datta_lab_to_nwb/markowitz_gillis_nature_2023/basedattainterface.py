"""Primary class for handling metadata non-specific to any other DataInterfaces."""

from pathlib import Path

import pandas as pd
import numpy as np
from neuroconv.basetemporalalignmentinterface import BaseTemporalAlignmentInterface
from neuroconv.utils import load_dict_from_file

DAT_IRES_CRE = "DAT-IRES-Cre (Slc6a3-IRES-Cre, The Jackson Laboratory 006660)"
DAT_IRES_CRE_AI32 = f"{DAT_IRES_CRE}::Ai32 (Rosa26-CAG-LSL-ChR2(H134R)-EYFP, The Jackson Laboratory 012569)"
WILD_TYPE = "wild type (C57BL/6J, The Jackson Laboratory 000664)"

# Lab group code (the `genotype` field of the subject metadata) -> genotype, opsin and description, per the Methods.
GROUP_CODE_TO_SUBJECT = {
    "snc-dls-ai32": dict(
        genotype=DAT_IRES_CRE_AI32,
        opsin="ChR2",
        description=(
            "Opto-DA mouse: DAT-IRES-Cre::Ai32, expressing ChR2 in dopamine neurons, with optical fibers over the "
            "dorsolateral striatum (DLS) for stimulating the axons of substantia nigra pars compacta (SNc) dopamine "
            "neurons."
        ),
    ),
    "snc-dls-ai32-jrcamp1a": dict(
        genotype=DAT_IRES_CRE_AI32,
        opsin="ChR2",
        description=(
            "Opto-DA mouse: DAT-IRES-Cre::Ai32, expressing ChR2 in dopamine neurons, with optical fibers over the "
            "dorsolateral striatum (DLS) for stimulating the axons of substantia nigra pars compacta (SNc) dopamine "
            "neurons. The lab group code also names jRCaMP1a."
        ),
    ),
    "snc-dls-ctrl": dict(
        genotype=DAT_IRES_CRE,
        opsin=None,
        description="No-opsin control mouse: DAT-IRES-Cre, with optical fibers over the dorsolateral striatum (DLS).",
    ),
    "vta-nacc-ctrl": dict(
        genotype=DAT_IRES_CRE,
        opsin=None,
        description="No-opsin control mouse: DAT-IRES-Cre, with optical fibers over the nucleus accumbens.",
    ),
    # Not described in the Methods; inferred from the lab group code, to be confirmed with the lab.
    "snc-eyfp": dict(
        genotype=f"{DAT_IRES_CRE} (inferred, not confirmed)",
        opsin=None,
        description=(
            "No-opsin control mouse. Presumed DAT-IRES-Cre with an eYFP control virus in the substantia nigra pars "
            "compacta (SNc); the paper does not describe this group, so the genotype and virus are inferred from the "
            "lab group code and not confirmed."
        ),
    ),
    "ctrl": dict(
        genotype=DAT_IRES_CRE,
        opsin=None,
        description="No-opsin control mouse: DAT-IRES-Cre.",
    ),
    "dls-dlight": dict(
        genotype=WILD_TYPE,
        opsin=None,
        description=(
            "C57BL/6J mouse with the dopamine sensor dLight1.1 (AAV) injected in the dorsolateral striatum (DLS) and "
            "an optical fiber over the injection site for photometry."
        ),
    ),
    "dms-dlight": dict(
        genotype=WILD_TYPE,
        opsin=None,
        description=(
            "C57BL/6J mouse with the dopamine sensor dLight1.1 (AAV) injected in the dorsomedial striatum (DMS) and "
            "an optical fiber over the injection site for photometry."
        ),
    ),
    "dls-chrimson-dlight": dict(
        genotype=DAT_IRES_CRE,
        opsin="ChrimsonR",
        description=(
            "DAT-IRES-Cre mouse with Cre-dependent ChrimsonR (AAV) injected in the right substantia nigra pars "
            "compacta (SNc) and dLight1.1 (AAV) injected in the right dorsolateral striatum (DLS), with an optical "
            "fiber over the DLS for photometry and for stimulating the ChrimsonR-expressing dopamine axons."
        ),
    ),
}


class BaseDattaInterface(BaseTemporalAlignmentInterface):
    """Base interface for markowitz_gillis_nature_2023 conversion w/ non-specific metadata"""

    def get_metadata(self) -> dict:
        metadata = super().get_metadata()
        session_metadata = load_dict_from_file(self.source_data["session_metadata_path"])
        session_metadata = session_metadata[self.source_data["session_uuid"]]
        subject_metadata = load_dict_from_file(self.source_data["subject_metadata_path"])
        subject_metadata = subject_metadata[session_metadata["subject_id"]]

        metadata["NWBFile"]["session_description"] = session_metadata["session_description"]
        metadata["NWBFile"]["session_start_time"] = session_metadata["session_start_time"]
        metadata["NWBFile"]["identifier"] = self.source_data["session_uuid"]
        metadata["NWBFile"]["session_id"] = self.source_data["session_id"]

        metadata["Subject"] = {}
        metadata["Subject"]["subject_id"] = session_metadata["subject_id"]
        metadata["Subject"]["sex"] = subject_metadata["sex"]
        group_code = subject_metadata["genotype"]
        subject = GROUP_CODE_TO_SUBJECT[group_code]
        metadata["Subject"]["genotype"] = subject["genotype"]
        metadata["Subject"]["description"] = f"{subject['description']} Lab group code: {group_code}."

        if self.source_data["alignment_path"] is not None and Path(self.source_data["alignment_path"]).exists():
            alignment_df = pd.read_parquet(self.source_data["alignment_path"])
            metadata["Alignment"]["slope"] = alignment_df["slope"].iloc[0]
            metadata["Alignment"]["bias"] = alignment_df["bias"].iloc[0]
        else:
            metadata["Alignment"]["slope"] = 1.0
            metadata["Alignment"]["bias"] = 0.0

        return metadata

    def get_metadata_schema(self) -> dict:
        metadata_schema = super().get_metadata_schema()
        if self.source_data["alignment_path"] is None:
            return metadata_schema
        metadata_schema["Alignment"] = {
            "type": "object",
            "description": "Metadata for temporal alignment with photometry data.",
            "required": True,
            "properties": {
                "slope": {
                    "description": "Slope of the linear regression mapping from behavioral video indices to demodulated photometry indices.",
                    "required": True,
                    "type": "float",
                },
                "bias": {
                    "description": "Bias of the linear regression mapping from behavioral video indices to demodulated photometry indices.",
                    "required": True,
                    "type": "float",
                },
                "start_time": {
                    "description": "Start time offset of raw fiber photometry data relative to behavioral video.",
                    "required": True,
                    "type": "float",
                },
            },
        }
        return metadata_schema

    def set_aligned_timestamps(self, aligned_timestamps: np.ndarray) -> None:
        self.aligned_timestamps = aligned_timestamps

    def get_timestamps(self) -> np.ndarray:
        return self.aligned_timestamps
