from data.modalities.base import ModalitySpec
from data.modalities.kinds import CompositionalTable

# Column name conventions: every column of a modality must fully match its pattern.
METAPHLAN_SPECIES = r"k__[^|]+\|p__[^|]+\|c__[^|]+\|o__[^|]+\|f__[^|]+\|g__[^|]+\|s__[^|]+"
"""MetaPhlAn species lineage, e.g. 'k__Bacteria|p__...|g__Prevotella|s__Prevotella_copri'."""

GTDB_SPECIES = r"d__[^;]+;p__[^;]+;c__[^;]+;o__[^;]+;f__[^;]+;g__[^;]+;s__[^;]+"
"""GTDB species lineage (sylph), e.g. 'd__Bacteria;p__...;g__Prevotella;s__Prevotella copri'."""

METACYC_PATHWAY = r"[^\s:]+: .+"
"""HUMAnN MetaCyc pathway, e.g. 'PWY-7221: guanosine ribonucleotides de novo biosynthesis'."""

TAXONOMY = ModalitySpec(
    name="taxonomy",
    kind=CompositionalTable(
        total=1.0,
        tolerance=1e-6,
        column_pattern=METAPHLAN_SPECIES,
    ),
    scope="per_sample",
    description=("Relative abundance of species (MetaPhlAn); each row sums to 1. "),
)

PATHWAYS = ModalitySpec(
    name="pathways",
    kind=CompositionalTable(
        total=10.0,
        tolerance=1e-6,
        column_pattern=METACYC_PATHWAY,
    ),
    scope="per_sample",
    description=("Relative abundance of metabolic pathways (HUMAnN), each row sums to 10. "),
)

TAXONOMY_METAPHLAN = ModalitySpec(
    name="taxonomy_metaphlan",
    kind=CompositionalTable(
        total=100.0,
        tolerance=1e-6,
        column_pattern=METAPHLAN_SPECIES,
    ),
    scope="per_sample",
    description="Species relative abundance from MetaPhlAn, in percent; columns are 'k__...|s__Genus_species'.",
)

TAXONOMY_SYLPH = ModalitySpec(
    name="taxonomy_sylph",
    kind=CompositionalTable(
        total=100.0,
        tolerance=1e-6,
        column_pattern=GTDB_SPECIES,
    ),
    scope="per_sample",
    description="Species relative abundance from sylph (GTDB taxonomy), in percent; columns are 'd__...;s__Genus species'.",
)

PATHWAYS_HUMANN = ModalitySpec(
    name="pathways_humann",
    kind=CompositionalTable(
        total=1.0,
        tolerance=1e-6,
        column_pattern=METACYC_PATHWAY,
    ),
    scope="per_sample",
    description="MetaCyc pathway relative abundance from HUMAnN; columns are 'PWY-ID: description'.",
)

type ModalityName = str
"""Name of a registered modality: a key of `MODALITY_SPECS` (e.g. "taxonomy_metaphlan")."""

MODALITY_SPECS: dict[ModalityName, ModalitySpec] = {
    spec.name: spec
    for spec in (TAXONOMY, PATHWAYS, TAXONOMY_METAPHLAN, TAXONOMY_SYLPH, PATHWAYS_HUMANN)
}


def get_modality_spec_from_registry(name: ModalityName) -> ModalitySpec:
    """Return the registered spec for a modality name."""
    return MODALITY_SPECS[name]
