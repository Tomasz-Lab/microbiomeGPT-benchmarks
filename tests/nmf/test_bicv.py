import numpy as np

import gut_former.nmf.run_bicv as bicv


def test_each_fold_is_a_partition_of_X(bicv_folds, synthetic_nmf):
    expected = np.sort(synthetic_nmf.X.ravel())
    assert len(bicv_folds) == 9

    for _idx, (A, B, C, D) in bicv_folds.items():
        assert A.shape[0] == B.shape[0]
        assert A.shape[1] == C.shape[1]
        assert D.shape == (C.shape[0], B.shape[1])

        got = np.sort(np.concatenate([b.ravel() for b in (A, B, C, D)]))
        np.testing.assert_array_equal(got, expected)


def test_every_cell_is_held_out_exactly_once(bicv_folds, synthetic_nmf):
    held_out = np.concatenate([A.ravel() for A, *_ in bicv_folds.values()])
    np.testing.assert_array_equal(np.sort(held_out), np.sort(synthetic_nmf.X.ravel()))


def test_run_cv_never_passes_heldout_block_to_nmf(bicv_folds, monkeypatch, tmp_path):
    A, B, C, D = bicv_folds[1]

    nmf_inputs = []

    def fake_nmf(X, n_components, **kwargs):
        X = np.asarray(X)
        nmf_inputs.append(X.copy())
        H = kwargs.get("H")
        if H is None:
            H = np.ones((n_components, X.shape[1]))
        W = np.ones((X.shape[0], n_components))
        return W, H, 0

    monkeypatch.setattr(bicv, "nmf", fake_nmf, raising=False)
    bicv.run_cv({1: (A, B, C, D)}, signatures=[3], outdir=tmp_path, n_runs=1)

    assert len(nmf_inputs) == 3
    for actual, expected in zip(nmf_inputs, (D, B, C.T), strict=True):
        np.testing.assert_array_equal(actual, expected)
