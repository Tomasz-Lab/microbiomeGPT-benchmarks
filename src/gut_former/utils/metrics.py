import numpy as np


def cos_sim(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    X_pred: np.ndarray[tuple[int, int], np.dtype[np.float64]],
):
    X_flat = np.concatenate(np.array(X))
    X_pred_flat = np.concatenate(np.array(X_pred))
    return X_pred_flat.dot(X_flat) / np.sqrt(X_pred_flat.dot(X_pred_flat) * X_flat.dot(X_flat))


def exp_var(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    X_pred: np.ndarray[tuple[int, int], np.dtype[np.float64]],
):
    return 1 - (np.sum((np.array(X) - np.array(X_pred)) ** 2) / np.sum(np.array(X) ** 2))


def rss(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    X_pred: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.float64:
    return np.sum((np.array(X) - np.array(X_pred)) ** 2)


def l2norm(
    X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    X_pred: np.ndarray[tuple[int, int], np.dtype[np.float64]],
):
    return np.sqrt(np.sum((np.array(X) - np.array(X_pred)) ** 2))
