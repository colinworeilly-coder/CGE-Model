import numpy as np
import pandas as pd
import pytest

from regcge.data.balance import balance_sam_cross_entropy, balance_sam_ras, ras
from regcge.data.sam import SAMError, check_balance, imbalances
from regcge.data.synthetic import split_accounts


def test_illustrative_sam_balanced(ds):
    assert check_balance(ds.sam, rel_tol=1e-12) < 1e-12


def test_ras_matches_targets():
    rng = np.random.default_rng(1)
    A = rng.uniform(0, 1, (6, 5)) * (rng.uniform(size=(6, 5)) > 0.2)
    r = rng.uniform(1, 2, 6)
    c = rng.uniform(1, 2, 5)
    c *= r.sum() / c.sum()
    X = ras(A, r, c)
    assert np.allclose(X.sum(1), r) and np.allclose(X.sum(0), c)
    assert np.all((A == 0) <= (X == 0))  # zeros preserved


def test_sam_ras_balances_perturbed():
    rng = np.random.default_rng(0)
    n = 8
    X = rng.uniform(1, 10, (n, n)) * (rng.uniform(size=(n, n)) > 0.3)
    np.fill_diagonal(X, 0)
    P = pd.DataFrame(X, index=[f"a{i}" for i in range(n)], columns=[f"a{i}" for i in range(n)])
    with pytest.raises(SAMError):
        check_balance(P)
    B = balance_sam_ras(P)
    assert check_balance(B, rel_tol=1e-8) < 1e-8
    assert ((P == 0) == (B == 0)).all().all()


def test_cross_entropy_keeps_signs_and_zeros(ds):
    rng = np.random.default_rng(2)
    P = ds.sam * np.exp(0.05 * rng.standard_normal(ds.sam.shape))
    B = balance_sam_cross_entropy(P)
    assert check_balance(B, rel_tol=1e-8) < 1e-8
    assert ((ds.sam == 0) == (B == 0)).all().all()
    assert (np.sign(B) == np.sign(ds.sam)).all().all()


def test_split_accounts_preserves_balance(ds):
    S2, mapping = split_accounts(ds.sam, {"MFG": 3, "H7": 2}, noise=0.0, seed=3)
    assert check_balance(S2, rel_tol=1e-12) < 1e-12
    assert mapping["MFG"] == ["MFG_1", "MFG_2", "MFG_3"]
    assert np.isclose(S2.to_numpy().sum(), ds.sam.to_numpy().sum())
