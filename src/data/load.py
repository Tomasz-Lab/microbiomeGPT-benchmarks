from data.modalities.base import Modality
from data.modataset.base import MultiOmicsDataset, MultiOmicsDatasetSpec


def load_dataset(spec: MultiOmicsDatasetSpec) -> MultiOmicsDataset:
    """Build a `MultiOmicsDataset` from its recipe."""
    modalities = {}
    for name, entry in spec.modalities.items():
        raw = entry.source.read()
        # The first column is the sample id, in the modality tables and in the metadata.
        raw = raw.set_index(raw.columns[0])
        data = entry.spec.validate(raw)
        modalities[name] = Modality(spec=entry.spec, data=data)

    raw_metadata = spec.metadata.source.read()
    metadata = raw_metadata.set_index(raw_metadata.columns[0])

    return MultiOmicsDataset(name=spec.name, modalities=modalities, metadata=metadata)
