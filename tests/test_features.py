import numpy as np

from phantomlink import features


def test_feature_values():
    f = features.extract("secure-login.pay123")
    assert f["length"] == 19
    assert f["n_labels"] == 2
    assert f["n_digits"] == 3
    assert f["n_hyphens"] == 1
    assert f["max_digit_run"] == 3
    assert f["suspicious_tokens"] >= 3   # secure, login, pay
    assert 0 < f["entropy"] < 5


def test_matrix_order_and_determinism():
    m1 = features.matrix(["abc", "a-1.b"])
    m2 = features.matrix(["abc", "a-1.b"])
    assert m1.shape == (2, len(features.FEATURE_NAMES))
    assert np.array_equal(m1, m2)
