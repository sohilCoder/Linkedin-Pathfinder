"""Text normalization for people, companies and profile URLs.

Different members' exports describe the same person and company in slightly
different ways ("Jon Smith, PhD" at "Meta Platforms, Inc." vs "Jonathan Smith"
at "Facebook"). Everything in this module turns those variants into a
canonical key so the matcher can compare like with like.
"""

from __future__ import annotations

import re
import unicodedata

# --- People -----------------------------------------------------------------

# Each group lists a formal first name followed by its common short forms.
# A nickname may belong to several groups ("alex" -> alexander / alexandra),
# so we map every name to the *set* of groups it belongs to.
_NICKNAME_GROUPS: list[tuple[str, ...]] = [
    ("jonathan", "jon", "jonny"),
    ("john", "johnny", "jack"),
    ("michael", "mike", "mikey", "mick"),
    ("robert", "rob", "bob", "bobby", "robbie"),
    ("elizabeth", "liz", "beth", "lizzie", "eliza"),
    ("katherine", "kate", "katie", "kathy", "kat"),
    ("catherine", "cate", "cathy", "kate", "katie"),
    ("william", "will", "bill", "billy", "liam"),
    ("alexander", "alex", "xander", "sasha"),
    ("alexandra", "alex", "lexi", "sasha"),
    ("christopher", "chris", "topher"),
    ("christina", "chris", "tina", "christy"),
    ("daniel", "dan", "danny"),
    ("benjamin", "ben", "benji"),
    ("samantha", "sam", "sammy"),
    ("samuel", "sam", "sammy"),
    ("matthew", "matt"),
    ("nicholas", "nick", "nicky"),
    ("jennifer", "jen", "jenny"),
    ("rebecca", "becca", "becky"),
    ("anthony", "tony"),
    ("joseph", "joe", "joey"),
    ("victoria", "tori", "vicky"),
    ("jessica", "jess", "jessie"),
    ("andrew", "andy", "drew"),
    ("timothy", "tim", "timmy"),
    ("stephanie", "steph"),
    ("patricia", "pat", "trish", "patty"),
    ("patrick", "pat"),
    ("abigail", "abby"),
    ("zachary", "zach", "zack"),
    ("margaret", "maggie", "meg", "peggy"),
    ("theodore", "theo", "ted", "teddy"),
    ("gabriel", "gabe"),
    ("natalie", "nat"),
    ("nathaniel", "nate", "nat"),
    ("joshua", "josh"),
    ("david", "dave"),
    ("richard", "rick", "rich", "dick"),
    ("thomas", "tom", "tommy"),
    ("edward", "ed", "eddie", "ted"),
    ("james", "jim", "jimmy", "jamie"),
    ("susan", "sue", "susie"),
    ("deborah", "deb", "debbie"),
]

_NAME_GROUPS: dict[str, set[int]] = {}
for _idx, _group in enumerate(_NICKNAME_GROUPS):
    for _name in _group:
        _NAME_GROUPS.setdefault(_name, set()).add(_idx)

# Tokens people put in name fields that are not part of the name.
_NAME_NOISE = {
    "phd", "md", "mba", "cpa", "pmp", "pe", "cfa", "jd", "ms", "msc", "bs",
    "ba", "rn", "dds", "esq", "jr", "sr", "ii", "iii", "iv", "mr", "mrs",
    "ms", "dr", "prof",
}


def strip_accents(text: str) -> str:
    """'José Müller' -> 'Jose Muller'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _name_tokens(text: str) -> list[str]:
    text = strip_accents(text or "").lower()
    text = re.sub(r"\(.*?\)", " ", text)          # "(she/her)", "(Bob)"
    text = text.replace("-", "").replace("'", "")  # "smith-jones", "o'brien"
    text = re.sub(r"[^a-z\s]", " ", text)          # emoji, commas, dots
    return [t for t in text.split() if t not in _NAME_NOISE]


def normalize_first_name(first: str) -> str:
    """First real token of the first-name field: 'Mary J.' -> 'mary'."""
    tokens = _name_tokens(first)
    return tokens[0] if tokens else ""


def normalize_last_name(last: str) -> str:
    """Last-name field without credentials or stray initials.

    'Doe, PhD' -> 'doe'; 'Q. Public' -> 'public'; 'Van Dyke' -> 'vandyke'.
    """
    tokens = _name_tokens(last)
    if len(tokens) > 1:
        tokens = [t for t in tokens if len(t) > 1] or tokens
    return "".join(tokens)


def first_names_compatible(a: str, b: str) -> bool:
    """Could these two normalized first names belong to the same person?

    True for exact matches, known nickname pairs ('jon' / 'jonathan') and a
    bare initial ('j') matching any name starting with that letter.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) == 1 or len(b) == 1:
        return a[0] == b[0]
    return bool(_NAME_GROUPS.get(a, set()) & _NAME_GROUPS.get(b, set()))


def is_initial(first: str) -> bool:
    return len(first) == 1


# --- Companies --------------------------------------------------------------

_LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "plc", "gmbh", "ag", "sa", "lp", "llp", "com",
}

# Canonical key for companies that are commonly written in unrelated ways.
# Keys are already-normalized strings (lowercase, punctuation removed).
_COMPANY_ALIASES = {
    "facebook": "meta",
    "meta platforms": "meta",
    "alphabet": "google",
    "amazon web services": "amazon",
    "aws": "amazon",
    "international business machines": "ibm",
    "jpmorgan": "jpmorgan chase",
    "j p morgan": "jpmorgan chase",
    "jp morgan": "jpmorgan chase",
    "jp morgan chase": "jpmorgan chase",
    "jpmorgan chase bank": "jpmorgan chase",
}


def normalize_company(name: str) -> str:
    """'Meta Platforms, Inc.' -> 'meta'; 'JPMorgan Chase & Co.' -> 'jpmorgan chase'."""
    text = strip_accents(name or "").lower().strip()
    if not text:
        return ""
    text = re.sub(r"\(.*?\)", " ", text)            # "Amazon Web Services (AWS)"
    text = re.sub(r"\s*(&|\band\b)\s*co\b\.?", " ", text)  # "& Co."
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    if tokens and tokens[0] == "the" and len(tokens) > 1:
        tokens = tokens[1:]
    while len(tokens) > 1 and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    key = " ".join(tokens)
    return _COMPANY_ALIASES.get(key, key)


# --- Profile URLs -----------------------------------------------------------

def normalize_url(url: str) -> str:
    """Canonical form of a LinkedIn profile URL, or '' if there is none.

    'https://www.linkedin.com/in/Priya-Sharma/?trk=x' -> 'linkedin.com/in/priya-sharma'
    """
    u = (url or "").strip().lower()
    if not u:
        return ""
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^(www\.|[a-z]{2}\.)", "", u)   # www. or country subdomain
    u = u.split("?")[0].split("#")[0].rstrip("/")
    match = re.match(r"linkedin\.com/in/([^/]+)", u)
    return f"linkedin.com/in/{match.group(1)}" if match else u
