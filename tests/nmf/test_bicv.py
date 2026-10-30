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


def test_every_cell_is_held_out_exactly_once(cell_id_matrix):
    X = cell_id_matrix
    seen = np.zeros_like(X)

    for A, *_ in bicv.make_folds(X, h=3):
        np.add.at(seen.ravel(), A.ravel(), 1)

    np.testing.assert_array_equal(seen, np.ones_like(X))


def test_B_shares_A_rows_and_C_shares_A_columns(cell_id_matrix):
    X = cell_id_matrix
    n_cols = X.shape[1]

    # map each value to coresonding column or row
    original_row_by_cell = np.repeat(np.arange(X.shape[0]), n_cols)
    original_column_by_cell = np.tile(np.arange(n_cols), X.shape[0])

    for A, B, C, _ in bicv.make_folds(X, h=3):
        # check to what rows values from A and B belongs
        a_rows = original_row_by_cell[A[:, 0]]
        b_rows = original_row_by_cell[B[:, 0]]
        # check to what columns values from A and C belongs
        a_columns = original_column_by_cell[A[0, :]]
        c_columns = original_column_by_cell[C[0, :]]

        np.testing.assert_array_equal(a_rows, b_rows)
        np.testing.assert_array_equal(a_columns, c_columns)


def test_evaluate_signature_keeps_heldout_block_out_of_nmf(bicv_folds, monkeypatch, nmf_params):
    A, B, C, D = bicv_folds[1]

    fitted_inputs = []
    fitted_bases = []
    transformed_inputs = []
    fixed_bases = []

    real_fit = bicv.NMF.fit
    real_transform = bicv.NMF.transform

    def fit_spy(model, X, *args, **kwargs):
        fitted_inputs.append(np.array(X, copy=True))
        fitted_model = real_fit(model, X, *args, **kwargs)
        fitted_bases.append(fitted_model.W_train_.T.copy())
        return fitted_model

    def transform_spy(model, X, H=None):
        transformed_inputs.append(np.array(X, copy=True))
        fixed_bases.append(None if H is None else np.array(H, copy=True))
        return real_transform(model, X, H=H)

    monkeypatch.setattr(bicv.NMF, "fit", fit_spy)
    monkeypatch.setattr(bicv.NMF, "transform", transform_spy)
    params = {**nmf_params, "n_runs": 1, "max_iter": 500}
    bicv.evaluate_signature(A, B, C, D, params=params, signature=3)

    assert len(fitted_inputs) == 1
    np.testing.assert_array_equal(fitted_inputs[0], D)
    assert len(transformed_inputs) == 2
    np.testing.assert_array_equal(transformed_inputs[0], B)
    np.testing.assert_array_equal(transformed_inputs[1], C.T)
    assert fixed_bases[0] is None
    np.testing.assert_array_equal(fixed_bases[1], fitted_bases[0])
