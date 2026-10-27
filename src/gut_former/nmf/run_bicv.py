import logging
import time
from itertools import product
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf

from gut_former.nmf.nmf_core import NMF
from gut_former.utils.log_config import setup_logging
from gut_former.utils.metrics import cos_sim, exp_var
from gut_former.utils.metrics import l2norm as calc_l2norm
from gut_former.utils.metrics import rss as calc_rss

log = logging.getLogger(__name__)


def make_folds(M: np.ndarray, h: int = 3):
    """
    Yield h*h (A, B, C, D) tuples; each fold has a different block as A.

    M is first thought of as an h x h grid of blocks (h=3 here). Blocks are
    numbered row by row:

            +---+---+---+
            | 1 | 2 | 3 |
            +---+---+---+
            | 4 | 5 | 6 |
            +---+---+---+
            | 7 | 8 | 9 |
            +---+---+---+

    Step 1: roll the matrix cyclically by whole blocks.
        np.roll(M, (r * row_shift, c * column_shift), axis=(0, 1))
        r = how many blocks to move rows down, c = how many blocks to move
        columns right. Whatever falls off one edge re-enters on the other.

        Example r=1, c=1:

          original         rows down by 1     columns right by 1
          +---+---+---+    +---+---+---+      +---+---+---+
          | 1 | 2 | 3 |    | 7 | 8 | 9 |      | 9 | 7 | 8 |
          +---+---+---+    +---+---+---+      +---+---+---+
          | 4 | 5 | 6 | -> | 1 | 2 | 3 |  ->  | 3 | 1 | 2 |
          +---+---+---+    +---+---+---+      +---+---+---+
          | 7 | 8 | 9 |    | 4 | 5 | 6 |      | 6 | 4 | 5 |
          +---+---+---+    +---+---+---+      +---+---+---+

        Whole block-rows and block-columns move together, so blocks that
        shared samples (a row) or pathways (a column) still do.

    Step 2: all h*h pairs (r, c) give every block exactly one turn in the
    top-left corner (A):

                    c=0    c=1    c=2
            r=0     A=1    A=3    A=2
            r=1     A=7    A=9    A=8
            r=2     A=4    A=6    A=5

    Step 3: cut the rolled matrix after the first block-row and block-column.
    Sizes are proportional (A is 1/3 x 1/3 of M, not a quarter):

              1 block   2 blocks
            +---------+--------------------+
            |    A    |         B          |  1 block
            +---------+--------------------+
            |         |                    |
            |    C    |         D          |  2 blocks
            |         |                    |
            +---------+--------------------+

        A - held-out block (1/9 of M), never used for training
        B - same samples as A, other pathways (2/9)
        C - same pathways as A, other samples (2/9)
        D - the rest (4/9); the model is trained here

        For the r=1, c=1 example above: A=9, B=[7 8], C=[3 / 6], D=[1 2 / 4 5].
    """
    rows, cols = M.shape

    for r, c in product(range(h), repeat=2):
        r_shift = r * rows // h
        c_shift = c * cols // h

        r_size = ((r + 1) * rows // h) - r_shift
        c_size = ((c + 1) * cols // h) - c_shift

        R = np.roll(M, (r_shift, c_shift), axis=(0, 1))

        A = R[:r_size, :c_size]
        B = R[:r_size, c_size:]
        C = R[r_size:, :c_size]
        D = R[r_size:, c_size:]

        yield A, B, C, D


def shuffle_rows_and_cols(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly permute rows (samples) and columns (pathways) independently."""
    return rng.permutation(rng.permutation(X, axis=0), axis=1)


def score(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "evar": exp_var(true, pred),
        "cos_sim": cos_sim(true, pred),
        "rss": calc_rss(true, pred),
        "l2norm": calc_l2norm(true, pred),
    }


def summarize_block_a(df: pd.DataFrame) -> pd.DataFrame:
    summary = df[df["block"] == "A"].groupby("signature")[["evar", "cos_sim"]].agg(["mean", "std"])
    summary.columns = ["evar_mean", "evar_std", "cos_sim_mean", "cos_sim_std"]
    return summary


def evaluate_signature(A, B, C, D, params: dict, signature: int) -> list[dict]:
    """
    Fit NMF on D, fold in W_a and H_a, and return one metrics row per block.

    Goal: measure how well a model with `signature` components reconstructs
    data it has never seen, so the best number of signatures can be chosen.
    Block A is the held-out test: it shares neither samples nor features with D.

    Layout of the fold:

                      features of A      features of D
                    +----------------+--------------------+
        samples     |                |                    |
        of A        |       A        |         B          |
                    |   (held out)   |                    |
                    +----------------+--------------------+
        samples     |                |                    |
        of D        |       C        |         D          |
                    |                |    (train here)    |
                    +----------------+--------------------+

    Every NMF factorization has the form  block ≈ W @ H, where
        W (samples x signatures)   : how much of each signature a sample has
        H (signatures x features)  : how each signature looks on the features
    Blocks in the same row share W, blocks in the same column share H:

        A ≈ W_a @ H_a        B ≈ W_a @ H_d
        C ≈ W_d @ H_a        D ≈ W_d @ H_d

    Steps:

        1. Train on D.
               D ≈ W_d @ H_d        -> learn W_d and H_d

        2. Find W_a from B.
               B ≈ W_a @ H_d        H_d is known, solve only for W_a
           B has the same samples as A, so W_a is also the W of A.

        3. Find H_a from C.
               C ≈ W_d @ H_a        W_d is known, solve only for H_a
           C has the same features as A, so H_a is also the H of A.
           The solver can only fix H and solve for W, so the equation is
           transposed:  C.T ≈ H_a.T @ W_d.T  (W_d.T plays the role of fixed H).

        4. Predict A.
               A_pred = W_a @ H_a
           W_a came only from B and H_a only from C, so A was never used.

                 D ──► W_d, H_d
                        │    │
              B ◄───────┘    └───────► C
              │ (fix H_d)     (fix W_d) │
              ▼                         ▼
             W_a                       H_a
              └──────────┬─────────────┘
                         ▼
                 A_pred = W_a @ H_a   vs   A  -> score

    Why not a plain train/test split on samples? There, W_test is fitted to
    X_test and the result is compared with the same X_test, so the error keeps
    falling as signatures are added and never shows overfitting. Here the data
    used for scoring (A) is never used to fit anything, so extra signatures
    that only fit noise stop helping and the error on A has a minimum.

    Only block A is a true held-out score. B and C are partly fitted (to get
    W_a and H_a) and D is the training block, so they are kept as diagnostics.
    """
    # Train on D only
    model = NMF(**{**params, "n_components": signature}).fit(D)
    W_d, H_d = model.W_train_, model.H_

    # B shares samples with A: B ≈ W_a @ H_d
    W_a = model.transform(B)

    # C shares features with A: C ≈ W_d @ H_a  (transposed: C.T ≈ H_a.T @ W_d.T)
    H_a = model.transform(C.T, H=W_d.T).T

    blocks = {
        "A": (A, W_a @ H_a),
        "B": (B, W_a @ H_d),
        "C": (C, W_d @ H_a),
        "D": (D, W_d @ H_d),
    }
    return [
        {"signature": signature, "block": block, **score(true, pred)}
        for block, (true, pred) in blocks.items()
    ]


def run_bicv_task(cfg: DictConfig, **_):
    setup_logging()
    log.info("Starting biCV script..")

    X = hydra.utils.instantiate(cfg.dataset)
    log.info("Data loaded | shape: %s", X.shape)

    output_path = Path(HydraConfig.get().runtime.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    signatures = list(range(cfg.task.n_components_min, cfg.task.n_components_max + 1))
    params = OmegaConf.to_container(cfg.model.parameters, resolve=True)
    log.info("Signatures to test: %s", signatures)

    M = shuffle_rows_and_cols(X.values, np.random.default_rng(seed=cfg.task.seed))

    def process_task(fold_idx, A, B, C, D, signature):
        rows = evaluate_signature(A, B, C, D, params, signature)
        return [{"fold": fold_idx, **row} for row in rows]

    t_start = time.time()

    results_nested = Parallel(n_jobs=cfg.task.n_jobs, verbose=10)(
        delayed(process_task)(fold_idx, A, B, C, D, signature)
        for fold_idx, (A, B, C, D) in enumerate(make_folds(M, h=cfg.task.h), start=1)
        for signature in signatures
    )

    results = [row for sublist in results_nested for row in sublist]

    log.info("BiCV done in %.0fs", time.time() - t_start)

    df = pd.DataFrame(results)
    df.to_csv(output_path / "bicv_nmf_stats.csv", index=False)

    summary = summarize_block_a(df)
    log.info("Block A summary (held-out validation):\n%s", summary.round(3).to_string())

    best = summary["evar_mean"].idxmax()
    log.info("Best signature: %d (evar_mean=%.3f)", best, summary.loc[best, "evar_mean"])
