from data.modalities.base import ModalitySpec
from data.modalities.kinds import CompositionalTable

TAXONOMY = ModalitySpec(
    name="taxonomy",
    kind=CompositionalTable(
        total=1.0,
        tolerance=1e-6,
        feature_pattern=r"k__[^|]+(\|[a-z]__[^|]+)*\|s__[^|]+",
        missing="fill_zero",
    ),
    scope="per_sample",
    description=(
        "Relative abundance of species (MetaPhlAn); each row sums to 1. "
        "Columns are full lineages 'k__...|...|s__<species>'."
    ),
)

PATHWAYS = ModalitySpec(
    name="pathways",
    kind=CompositionalTable(
        total=10.0,
        tolerance=1e-6,
        feature_pattern=r"[^:\s]+: .+",
        missing="fill_zero",
    ),
    scope="per_sample",
    description=(
        "Relative abundance of metabolic pathways (HUMAnN), each row sums to 10. "
        "Columns are '<MetaCyc ID>: <pathway name>'."
    ),
)
