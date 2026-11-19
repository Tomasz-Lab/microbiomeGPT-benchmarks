from data.modalities.base import Modality
from data.modalities.registry import get_modality_spec_from_registry
from data.modataset.base import MultiOmicsDataset, MultiOmicsDatasetSpec


def load_dataset(dataset_spec: MultiOmicsDatasetSpec) -> MultiOmicsDataset:
    """Build a `MultiOmicsDataset` from its recipe."""
    modalities = {}
    for name, source in dataset_spec.modalities.items():
        modality_spec = get_modality_spec_from_registry(name)
        raw = source.read()
        # The first column is the sample id, in the modality tables and in the metadata.
        raw = raw.set_index(raw.columns[0])
        data = modality_spec.validate(raw)
        modalities[name] = Modality(spec=modality_spec, data=data)

    raw_metadata = dataset_spec.metadata.source.read()
    metadata = raw_metadata.set_index(raw_metadata.columns[0])

    return MultiOmicsDataset(name=dataset_spec.name, modalities=modalities, metadata=metadata)
