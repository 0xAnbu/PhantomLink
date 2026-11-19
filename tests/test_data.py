import pytest

from phantomlink import data, paths


def test_www_and_suffix_are_removed():
    n = data.normalize_host("WWW.Secure-Login.Example.co.uk.")
    assert n.name == "secure-login.example"
    assert n.registered == "example.co.uk"
    assert n.suffix == "co.uk"


def test_port_and_ip_hosts():
    assert data.normalize_host("shop.example.com:8080").name == "shop.example"
    assert data.normalize_host("192.168.1.20") is None
    assert data.normalize_host("nosuffix") is None


def test_registered_domain_groups_subdomains():
    a = data.normalize_host("a.login.example.com")
    b = data.normalize_host("www.example.com")
    assert a.registered == b.registered == "example.com"
    assert data.split_of(a.registered) == data.split_of(b.registered)


def test_split_is_deterministic_and_balanced():
    names = [f"site{i}.com" for i in range(20000)]
    splits = [data.split_of(n) for n in names]
    assert splits == [data.split_of(n) for n in names]
    frac = splits.count("train") / len(splits)
    assert 0.67 < frac < 0.73


@pytest.mark.parametrize("name,ok", [
    ("secure-login", True), ("a.b-c.d1", True), ("-bad", False), ("bad-", False),
    ("a..b", False), ("", False), ("x" * 64, False), ("ok_" , False)])
def test_valid_name(name, ok):
    assert data.is_valid_name(name) is ok


@pytest.mark.skipif(not paths.PROCESSED_CSV.exists(), reason="processed dataset not built")
def test_processed_dataset_has_no_leakage_and_correct_labels():
    import pandas as pd
    df = pd.read_csv(paths.PROCESSED_CSV, keep_default_na=False)
    assert (df.groupby("registered")["split"].nunique() == 1).all()
    assert not df["name"].duplicated().any()
    assert not df["name"].str.startswith("www.").any()
    raw = paths.RAW_CSV
    if raw.exists():
        r = pd.read_csv(raw, usecols=["Domain", "label"], nrows=5000)
        # In PhiUSIIL, label 1 = legitimate. Check a known-direction statistic: legitimate
        # hosts in the raw file always start with www.
        assert r.loc[r.label == 1, "Domain"].str.lower().str.startswith("www.").all()


def test_repeated_www_is_removed():
    assert data.normalize_host("www.www.woodandthimble.com").name == "woodandthimble"
