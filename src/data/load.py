from data.modalities.base import Modality
from data.modataset.base import MultiOmicsDataset, MultiOmicsDatasetSpec


def load_dataset(spec: MultiOmicsDatasetSpec) -> MultiOmicsDataset:
    """Build a `MultiOmicsDataset` from its recipe."""
    modalities = {}
    for name, entry in spec.modalities.items():
        raw = entry.source.read()
        data = entry.spec.validate(raw)
        modalities[name] = Modality(spec=entry.spec, data=data)

    metadata = spec.metadata.source.read()

    return MultiOmicsDataset(name=spec.name, modalities=modalities, metadata=metadata)
