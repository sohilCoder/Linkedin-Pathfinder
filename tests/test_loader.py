from datetime import date

import pytest

from pathfinder.loader import ExportFormatError, MemberProfile, parse_connections_csv

REAL_SHAPE = '''﻿Notes:
"When exporting your connection data, you may notice that some of the email addresses are missing. You will only see email addresses for connections who have allowed their connections to see or download it."

First Name,Last Name,URL,Email Address,Company,Position,Connected On
Priya,Sharma,https://www.linkedin.com/in/priya-sharma,,LinkedIn,Software Engineer,15 Oct 2024
,,,,,,
"Marcus","O'Brien, PhD",,marcus@example.com,"Stripe, Inc.",Staff Engineer,02 Mar 2019
'''


def test_parses_real_export_shape():
    rows = parse_connections_csv(REAL_SHAPE)
    assert len(rows) == 2                      # blank (deactivated) row skipped
    assert rows[0].first_name == "Priya"
    assert rows[0].connected_on == date(2024, 10, 15)
    assert rows[1].company == "Stripe, Inc."   # quoted comma preserved
    assert rows[1].row_index == 1


def test_rejects_files_that_are_not_exports():
    with pytest.raises(ExportFormatError):
        parse_connections_csv("name,email\nPriya,p@example.com\n")


def test_unknown_dates_become_none():
    rows = parse_connections_csv("First Name,Last Name,Connected On\nA,B,someday\n")
    assert rows[0].connected_on is None


def test_profile_validation():
    p = MemberProfile.from_dict({"first_name": "José", "last_name": "Núñez", "past_companies": "Meta, Stripe"})
    assert p.member_id == "jose-nunez"
    assert p.past_companies == ["Meta", "Stripe"]
    with pytest.raises(ValueError):
        MemberProfile.from_dict({"first_name": "A", "last_name": "B", "member_id": "../etc"})
    with pytest.raises(ValueError):
        MemberProfile.from_dict({"first_name": "", "last_name": "B"})


def test_decode_bytes_handles_utf8_bom_and_cp1252():
    from pathfinder.loader import decode_bytes

    assert decode_bytes("José".encode("utf-8")) == "José"
    assert decode_bytes("﻿José".encode("utf-8")) == "José"
    assert decode_bytes("José".encode("cp1252")) == "José"
