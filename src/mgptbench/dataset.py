import pandas as pd
import torch
from torch.utils.data import Dataset


class BacteriaDataset(Dataset):
    def __init__(self, taxonomy: pd.DataFrame, pathways: pd.DataFrame) -> None:
        self.taxonomy: torch.Tensor = torch.tensor(taxonomy.values).float().unsqueeze(-1)
        self.pathways: torch.Tensor = torch.tensor(pathways.values).float().unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.taxonomy)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.taxonomy[idx], self.pathways[idx]
