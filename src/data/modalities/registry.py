from data.modalities.base import ModalitySpec
from data.modalities.kinds import CompositionalTable

TAXONOMY = ModalitySpec(
    name="taxonomy",
    kind=CompositionalTable(
        total=1.0,
        tolerance=1e-6,
        missing="fill_zero",
    ),
    scope="per_sample",
    description=("Relative abundance of species (MetaPhlAn); each row sums to 1. "),
)

PATHWAYS = ModalitySpec(
    name="pathways",
    kind=CompositionalTable(
        total=10.0,
        tolerance=1e-6,
        missing="fill_zero",
    ),
    scope="per_sample",
    description=("Relative abundance of metabolic pathways (HUMAnN), each row sums to 10. "),
)

TAXONOMY_METAPHLAN = ModalitySpec(
    name="taxonomy_metaphlan",
    kind=CompositionalTable(total=100.0, tolerance=1e-6, missing="fill_zero"),
    scope="per_sample",
    description="Species relative abundance from MetaPhlAn, in percent; columns are 'k__...|s__Genus_species'.",
)

TAXONOMY_SYLPH = ModalitySpec(
    name="taxonomy_sylph",
    kind=CompositionalTable(total=100.0, tolerance=1e-6, missing="fill_zero"),
    scope="per_sample",
    description="Species relative abundance from sylph (GTDB taxonomy), in percent; columns are 'd__...;s__Genus species'.",
)

PATHWAYS_HUMANN = ModalitySpec(
    name="pathways_humann",
    kind=CompositionalTable(total=1.0, tolerance=1e-6, missing="fill_zero"),
    scope="per_sample",
    description="MetaCyc pathway relative abundance from HUMAnN; columns are 'PWY-ID: description'.",
)

MODALITY_SPECS = {
    spec.name: spec
    for spec in (TAXONOMY, PATHWAYS, TAXONOMY_METAPHLAN, TAXONOMY_SYLPH, PATHWAYS_HUMANN)
}
