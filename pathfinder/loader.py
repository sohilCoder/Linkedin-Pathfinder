"""Reading LinkedIn "Connections.csv" exports and member profiles.

A real export (Settings > Data privacy > Get a copy of your data) starts with
a few "Notes:" lines before the actual header, so the parser looks for the
header row instead of assuming it is the first line.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

EXPORT_COLUMNS = [
    "First Name", "Last Name", "URL", "Email Address",
    "Company", "Position", "Connected On",
]

_DATE_FORMATS = ("%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y")

MEMBER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class ExportFormatError(ValueError):
    """The uploaded file does not look like a LinkedIn connections export."""


@dataclass(frozen=True)
class ConnectionRow:
    """One line of someone's Connections export."""

    row_index: int
    first_name: str
    last_name: str
    url: str
    email: str
    company: str
    position: str
    connected_on: date | None


@dataclass
class MemberProfile:
    """What a member tells us about themselves when they join.

    The export only lists *other* people, so school and past employers come
    from here. They power the "former colleagues" and "same school" signals.
    """

    member_id: str
    first_name: str
    last_name: str
    url: str = ""
    company: str = ""
    position: str = ""
    school: str = ""
    past_companies: list[str] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemberProfile":
        past = data.get("past_companies") or []
        if isinstance(past, str):
            past = [p.strip() for p in past.split(",")]
        profile = cls(
            member_id=str(data.get("member_id") or "").strip(),
            first_name=str(data.get("first_name") or "").strip(),
            last_name=str(data.get("last_name") or "").strip(),
            url=str(data.get("url") or "").strip(),
            company=str(data.get("company") or "").strip(),
            position=str(data.get("position") or "").strip(),
            school=str(data.get("school") or "").strip(),
            past_companies=[p for p in (str(x).strip() for x in past) if p],
        )
        if not profile.member_id:
            profile.member_id = make_member_id(profile.first_name, profile.last_name)
        if not MEMBER_ID_PATTERN.match(profile.member_id):
            raise ValueError(f"invalid member id: {profile.member_id!r}")
        if not profile.first_name or not profile.last_name:
            raise ValueError("a member needs a first and last name")
        return profile


def make_member_id(first: str, last: str) -> str:
    from .normalize import strip_accents

    slug = strip_accents(f"{first} {last}").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug[:64] or "member"


def read_text_file(path) -> str:
    """Read a text file whatever encoding it was saved in.

    Exports from LinkedIn are UTF-8, but a file opened and re-saved in Excel
    on Windows is often cp1252. Try UTF-8 (with or without a byte-order
    mark) first, then fall back to cp1252 instead of crashing.
    """
    from pathlib import Path

    return decode_bytes(Path(path).read_bytes())


def decode_bytes(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def parse_date(text: str) -> date | None:
    text = (text or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_connections_csv(text: str) -> list[ConnectionRow]:
    """Parse the text of a Connections.csv export into rows.

    Tolerates the "Notes:" preamble, a byte-order mark, blank lines and
    unknown extra columns. Raises ExportFormatError if no header is found.
    """
    text = text.lstrip("﻿")
    reader = csv.reader(io.StringIO(text))
    header: list[str] | None = None
    rows: list[ConnectionRow] = []

    for raw in reader:
        cells = [c.strip() for c in raw]
        if header is None:
            if "First Name" in cells and "Last Name" in cells:
                header = cells
            continue
        if not any(cells):
            continue
        record = dict(zip(header, cells))
        first = record.get("First Name", "")
        last = record.get("Last Name", "")
        if not first and not last:
            continue  # LinkedIn leaves deactivated accounts blank
        rows.append(
            ConnectionRow(
                row_index=len(rows),
                first_name=first,
                last_name=last,
                url=record.get("URL", ""),
                email=record.get("Email Address", ""),
                company=record.get("Company", ""),
                position=record.get("Position", ""),
                connected_on=parse_date(record.get("Connected On", "")),
            )
        )

    if header is None:
        raise ExportFormatError(
            "No 'First Name, Last Name, ...' header found. "
            "Upload the Connections.csv file from your LinkedIn data export."
        )
    return rows


def write_connections_csv(rows: list[dict], *, with_notes: bool = True) -> str:
    """Write rows in the same shape as a real LinkedIn export."""
    out = io.StringIO()
    if with_notes:
        out.write("Notes:\n")
        out.write(
            '"When exporting your connection data, you may notice that some of '
            "the email addresses are missing. You will only see email addresses "
            'for connections who have allowed their connections to see or download it."\n'
        )
        out.write("\n")
    writer = csv.DictWriter(out, fieldnames=EXPORT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in EXPORT_COLUMNS})
    return out.getvalue()
