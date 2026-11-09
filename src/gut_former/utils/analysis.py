import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from skbio.stats.ordination import pcoa

from gut_former.utils.types import Matrix


def _to_pcoa(arr: Matrix) -> pd.DataFrame:
    """Compute PCoA coordinates from a 2D array."""
    return pcoa(squareform(pdist(arr))).samples


def get_pcoa_coords(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Combine true and predicted into one PCoA space for visualization."""
    true_arr = true_df.values.clip(min=0)
    pred_arr = pred_df.values.clip(min=0)
    return _to_pcoa(np.vstack([true_arr, pred_arr]))


def get_disparity(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    """Procrustes disparity between separately computed PCoAs."""
    true_arr = true_df.values.clip(min=0)
    pred_arr = pred_df.values.clip(min=0)
    _, _, disparity = procrustes(_to_pcoa(true_arr), _to_pcoa(pred_arr))
    return float(disparity)
