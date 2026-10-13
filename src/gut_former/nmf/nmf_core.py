import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import non_negative_factorization

from gut_former.utils.metrics import cos_sim, exp_var

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NMFConfig:
    n_components: int
    init: str = "nndsvdar"
    solver: str = "mu"
    beta_loss: str = "kullback-leibler"
    max_iter: int = 2000
    n_runs: int = 100
    n_jobs: int = -2
    selection_metric: Literal["exp_var", "cos_sim", "composite"] = "exp_var"


@dataclass(frozen=True)
class NMFRunResult:
    seed: int
    exp_var_train: float
    cos_sim_train: float
    exp_var_val: float
    cos_sim_val: float
    W_train: np.ndarray
    W_val: np.ndarray
    H: np.ndarray


class NMF(BaseEstimator, TransformerMixin):
    def __init__(self, config: NMFConfig):
        if not isinstance(config, NMFConfig):
            raise TypeError(
                f"Expected NMFConfig, received: {type(config).__name__}"
            )

        self.config = config

        self.n_components = config.n_components
        self.init = config.init
        self.solver = config.solver
        self.beta_loss = config.beta_loss
        self.max_iter = config.max_iter
        self.n_runs = config.n_runs
        self.n_jobs = config.n_jobs
        self.selection_metric = config.selection_metric

        self.components_: Optional[np.ndarray] = None
        self.W_train_: Optional[np.ndarray] = None
        self.W_val_: Optional[np.ndarray] = None
        self.best_seed_: Optional[int] = None
        self.history_: List[Dict[str, float]] = []

    def _single_nmf_run(
        self, X_train: np.ndarray, X_val: np.ndarray, seed: int
    ) -> NMFRunResult:
        W_train, H, _ = non_negative_factorization(
            X_train,
            n_components=self.n_components,
            init=self.init,
            solver=self.solver,
            beta_loss=self.beta_loss,
            max_iter=self.max_iter,
            random_state=seed,
        )
        W_val, _, _ = non_negative_factorization(
            X_val,
            n_components=self.n_components,
            init="custom",
            solver=self.solver,
            beta_loss=self.beta_loss,
            max_iter=self.max_iter,
            update_H=False,
            H=H,
        )

        train_pred = W_train @ H
        val_pred = W_val @ H

        return NMFRunResult(
            seed=seed,
            exp_var_train=exp_var(X_train, train_pred),
            cos_sim_train=cos_sim(X_train, train_pred),
            exp_var_val=exp_var(X_val, val_pred),
            cos_sim_val=cos_sim(X_val, val_pred),
            W_train=W_train,
            W_val=W_val,
            H=H,
        )

    def _score(self, run: NMFRunResult) -> float:
        if self.selection_metric == "exp_var":
            return run.exp_var_val
        elif self.selection_metric == "cos_sim":
            return run.cos_sim_val
        elif self.selection_metric == "composite":
            return run.exp_var_val + run.cos_sim_val
        raise ValueError(f"Unknown selection metric: {self.selection_metric}")

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y=None,
    ):
        X_tr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_v = X_val.values if isinstance(X_val, pd.DataFrame) else X_val

        results: List[NMFRunResult] = Parallel(n_jobs=self.n_jobs)(
            delayed(self._single_nmf_run)(X_tr, X_v, seed)
            for seed in range(1, self.n_runs + 1)
        )

        valid_results = [r for r in results if np.isfinite(self._score(r))]

        if not valid_results:
            raise RuntimeError(
                "All NMF runs returned invalid values (NaN/Inf)."
            )

        best_run = max(valid_results, key=self._score)

        self.components_ = best_run.H
        self.W_train_ = best_run.W_train
        self.W_val_ = best_run.W_val
        self.best_seed_ = best_run.seed

        self.history_ = [
            {
                "seed": r.seed,
                "exp_var_train": r.exp_var_train,
                "cos_sim_train": r.cos_sim_train,
                "exp_var_val": r.exp_var_val,
                "cos_sim_val": r.cos_sim_val,
            }
            for r in results
        ]

        log.info(
            "Best run (seed %d) | exp_var_val: %.4f | cos_sim_val: %.4f",
            best_run.seed,
            best_run.exp_var_val,
            best_run.cos_sim_val,
        )
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError(
                "Model is not yet fitted. Call .fit() first."
            )

        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        W, _, _ = non_negative_factorization(
            X_arr,
            n_components=self.n_components,
            init="custom",
            solver=self.solver,
            beta_loss=self.beta_loss,
            max_iter=self.max_iter,
            update_H=False,
            H=self.components_,
        )
        return W