import pytest

from pathfinder.normalize import (
    first_names_compatible,
    normalize_company,
    normalize_first_name,
    normalize_last_name,
    normalize_url,
)


@pytest.mark.parametrize("raw, expected", [
    ("Meta Platforms, Inc.", "meta"),
    ("Facebook", "meta"),
    ("Amazon Web Services (AWS)", "amazon"),
    ("Amazon.com", "amazon"),
    ("JPMorgan Chase & Co.", "jpmorgan chase"),
    ("J.P. Morgan", "jpmorgan chase"),
    ("LinkedIn Corporation", "linkedin"),
    ("Google LLC", "google"),
    ("The Home Depot", "home depot"),
    ("Procter & Gamble", "procter and gamble"),
    ("", ""),
])
def test_company_normalization(raw, expected):
    assert normalize_company(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Doe, PhD", "doe"),
    ("Smith MBA", "smith"),
    ("O'Brien", "obrien"),
    ("Smith-Jones", "smithjones"),
    ("Van Dyke", "vandyke"),
    ("Müller", "muller"),
    ("Lee (she/her)", "lee"),
    ("Q. Public", "public"),
])
def test_last_name_normalization(raw, expected):
    assert normalize_last_name(raw) == expected


def test_first_name_takes_first_token():
    assert normalize_first_name("Mary J.") == "mary"
    assert normalize_first_name("José") == "jose"
    assert normalize_first_name("J.") == "j"


@pytest.mark.parametrize("a, b, ok", [
    ("jon", "jonathan", True),
    ("bob", "robert", True),
    ("j", "jonathan", True),        # initial
    ("alex", "alexandra", True),
    ("john", "jonathan", False),    # different names, not nicknames
    ("jane", "john", False),
    ("j", "kate", False),
    ("", "kate", False),
])
def test_first_name_compatibility(a, b, ok):
    assert first_names_compatible(a, b) is ok
    assert first_names_compatible(b, a) is ok


def test_url_normalization():
    assert normalize_url("https://www.linkedin.com/in/Priya-Sharma/?trk=abc") == "linkedin.com/in/priya-sharma"
    assert normalize_url("http://uk.linkedin.com/in/priya-sharma") == "linkedin.com/in/priya-sharma"
    assert normalize_url("") == ""
