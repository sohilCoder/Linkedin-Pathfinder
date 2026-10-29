"""Generate a realistic *fake* professional network for development and demos.

The output has exactly the same layout the app stores real uploads in, and
each member's Connections.csv mimics a real LinkedIn export (including the
"Notes:" preamble and date format). Swapping in real exports later needs no
code changes.

What makes it realistic rather than random:
  * Heavy-tailed degrees: a few people have hundreds of connections, most
    have a few dozen.
  * Clustering: people mostly connect with coworkers (past and present) and
    classmates from nearby graduation years.
  * Careers: people change jobs, so former teammates end up spread across
    companies, which is what creates interesting intro paths.
  * Deliberate mess: nicknames, initials, credentials, pronouns, accents
    dropped, company spelling variants, missing URLs, and a few genuinely
    different people who share a name and employer.

Every mention's true identity is written to ground_truth.csv, so the matcher
can be scored exactly (see `python -m pathfinder eval`).

Note: the generator's nickname list deliberately includes a few forms the
matcher does not know (e.g. "Kit", "Mags"), so evaluation is not rigged.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import shutil
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from .loader import MemberProfile, make_member_id, write_connections_csv
from .normalize import strip_accents

# formal name -> short forms people actually go by (independent of the
# matcher's table on purpose; "Kit", "Mags", "Zeke" etc. are unknown to it)
FIRST_NAMES: dict[str, list[str]] = {
    "Jonathan": ["Jon"], "Michael": ["Mike"], "Robert": ["Rob", "Bob"],
    "Elizabeth": ["Liz", "Beth", "Libby"], "Katherine": ["Kate", "Katie", "Kit"],
    "William": ["Will", "Bill"], "Alexander": ["Alex"], "Alexandra": ["Alex", "Lexi"],
    "Christopher": ["Chris"], "Daniel": ["Dan"], "Benjamin": ["Ben"],
    "Samantha": ["Sam"], "Matthew": ["Matt"], "Nicholas": ["Nick"],
    "Jennifer": ["Jen"], "Rebecca": ["Becca"], "Anthony": ["Tony"],
    "Joseph": ["Joe"], "Victoria": ["Tori"], "Jessica": ["Jess"],
    "Andrew": ["Andy", "Drew"], "Timothy": ["Tim"], "Stephanie": ["Steph"],
    "Abigail": ["Abby"], "Zachary": ["Zach"], "Margaret": ["Maggie", "Mags"],
    "Theodore": ["Theo"], "Gabriel": ["Gabe"], "Natalie": ["Nat"],
    "Ezekiel": ["Zeke"], "Joshua": ["Josh"], "David": ["Dave"],
    "Priya": [], "Marcus": [], "Wei": [], "Aisha": [], "Diego": [],
    "Fatima": [], "Hiroshi": [], "Olu": [], "Sofia": [], "Arjun": [],
    "Mei": [], "Omar": [], "Lucia": [], "Kwame": [], "Ananya": [],
    "Mateo": [], "Yuki": [], "Chloe": [], "Ethan": [], "Zara": [],
    "Ravi": [], "Elena": [], "Jamal": [], "Ingrid": [], "Noah": [],
    "Leila": [], "Tariq": [], "Hana": [], "Bruno": [], "Ines": [],
    "José": [], "Zoë": [], "Renée": [], "Björn": [], "Amara": [],
    "Kenji": [], "Nadia": [], "Sven": [], "Imani": [], "Rohan": [],
}

LAST_NAMES = [
    "Sharma", "Chen", "Johnson", "Garcia", "Nguyen", "Okafor", "Smith",
    "Patel", "Kim", "Williams", "Brown", "Rodriguez", "Martinez", "Lee",
    "Tanaka", "Müller", "O'Brien", "Smith-Jones", "Van Dyke", "Haddad",
    "Kowalski", "Rossi", "Silva", "Ivanova", "Mensah", "Andersson", "Cohen",
    "Wright", "Lopez", "Gupta", "Hernandez", "Park", "Singh", "Ali",
    "Novak", "Dubois", "Schmidt", "Yamamoto", "Adeyemi", "Fischer",
    "Bennett", "Reyes", "Morales", "Walsh", "Ortiz", "Ramos", "Khan",
    "Liu", "Zhang", "Wang", "Das", "Iyer", "Fernández", "Jensen",
    "Baker", "Carter", "Murphy", "Rivera", "Cooper", "Price", "Hughes",
    "Foster", "Bailey", "Diaz", "Ward", "Torres", "Peterson", "Gray",
]

# canonical display name -> spellings that show up in exports
COMPANIES: dict[str, list[str]] = {
    "LinkedIn": ["LinkedIn", "Linkedin", "LinkedIn Corporation"],
    "Meta": ["Meta", "Facebook", "Meta Platforms, Inc."],
    "Google": ["Google", "Google LLC", "Alphabet"],
    "Microsoft": ["Microsoft", "Microsoft Corporation"],
    "Amazon": ["Amazon", "Amazon Web Services (AWS)", "Amazon.com"],
    "Apple": ["Apple", "Apple Inc."],
    "Stripe": ["Stripe", "Stripe, Inc."],
    "Salesforce": ["Salesforce", "salesforce.com"],
    "Adobe": ["Adobe", "Adobe Inc."],
    "NVIDIA": ["NVIDIA", "Nvidia Corporation"],
    "Netflix": ["Netflix"],
    "Uber": ["Uber", "Uber Technologies"],
    "Airbnb": ["Airbnb", "Airbnb, Inc."],
    "Databricks": ["Databricks"],
    "Snowflake": ["Snowflake", "Snowflake Inc."],
    "Capital One": ["Capital One"],
    "JPMorgan Chase": ["JPMorgan Chase & Co.", "J.P. Morgan", "JPMorgan Chase"],
    "Deloitte": ["Deloitte"],
    "Accenture": ["Accenture"],
    "IBM": ["IBM", "International Business Machines"],
    "Caterpillar": ["Caterpillar", "Caterpillar Inc."],
    "State Farm": ["State Farm"],
    "Epic Systems": ["Epic Systems", "Epic"],
    "Intel": ["Intel", "Intel Corporation"],
    "Qualcomm": ["Qualcomm"],
    "Workday": ["Workday"],
    "Atlassian": ["Atlassian"],
    "Kestrel AI": ["Kestrel AI", "Kestrel AI, Inc."],
    "Prairie Analytics": ["Prairie Analytics"],
    "Foxglove Health": ["Foxglove Health"],
    "Lumen Robotics": ["Lumen Robotics"],
    "Tidewater Labs": ["Tidewater Labs"],
}

# rough relative size of each employer in the simulated world
_COMPANY_WEIGHT = {
    "Microsoft": 9, "Amazon": 10, "Google": 9, "Meta": 6, "LinkedIn": 5,
    "Apple": 5, "Deloitte": 6, "Accenture": 6, "JPMorgan Chase": 6,
    "IBM": 5, "Capital One": 4, "State Farm": 4, "Caterpillar": 4,
    "Salesforce": 4, "Intel": 4, "Epic Systems": 3,
}

SCHOOLS = [
    "Midwest State University", "Lakeshore University", "Pacific Tech",
    "Northgate College", "Riverside University",
    "Eastern Institute of Technology", "Summit State", "Bayview University",
]
HOME_SCHOOL = SCHOOLS[0]   # the simulated student club lives here

TITLES_BY_LEVEL = [
    ["Software Engineer Intern", "Data Science Intern", "Product Management Intern"],
    ["Software Engineer", "Data Analyst", "Associate Product Manager", "Technical Recruiter"],
    ["Software Engineer II", "Data Scientist", "Product Manager", "Machine Learning Engineer", "Product Designer"],
    ["Senior Software Engineer", "Senior Data Scientist", "Senior Product Manager", "Site Reliability Engineer"],
    ["Staff Software Engineer", "Engineering Manager", "Principal Engineer", "Director of Engineering"],
]

AS_OF = date(2026, 10, 1)

_NOTES_CREDENTIALS = [", PhD", " MBA", ", PMP", " (she/her)", " (he/him)", ", CPA"]


@dataclass
class Person:
    pid: int
    first: str
    last: str
    school: str
    grad_year: int
    career: list[str] = field(default_factory=list)   # canonical company names
    position: str = ""
    slug: str = ""

    @property
    def is_student(self) -> bool:
        return self.grad_year > AS_OF.year

    @property
    def current_company(self) -> str:
        return self.career[-1] if self.career else self.school


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", strip_accents(text).lower()).strip("-")


class _World:
    def __init__(self, n_people: int, rng: random.Random):
        self.rng = rng
        self.people: list[Person] = []
        self.edges: dict[tuple[int, int], date] = {}
        self.adj: dict[int, set[int]] = {}
        self._make_people(n_people)

    # -- people ---------------------------------------------------------
    def _make_people(self, n: int) -> None:
        rng = self.rng
        companies = list(COMPANIES)
        weights = [_COMPANY_WEIGHT.get(c, 2) for c in companies]
        used_slugs: set[str] = set()
        for pid in range(n):
            first = rng.choice(list(FIRST_NAMES))
            last = rng.choice(LAST_NAMES)
            # home-school students and recent grads are over-represented,
            # like a real student's network
            school = HOME_SCHOOL if rng.random() < 0.35 else rng.choice(SCHOOLS)
            grad_year = rng.randint(1998, AS_OF.year + 3)
            p = Person(pid, first, last, school, grad_year)
            years = AS_OF.year - grad_year
            if years >= 0:
                n_jobs = min(4, 1 + years // 3 + (rng.random() < 0.3))
                for _ in range(n_jobs):
                    c = rng.choices(companies, weights)[0]
                    if not p.career or p.career[-1] != c:
                        p.career.append(c)
                level = min(4, 1 + years // 3)
                p.position = rng.choice(TITLES_BY_LEVEL[level])
            else:
                p.position = "Student"
                if rng.random() < 0.3:   # an internship already done
                    p.career.append(rng.choices(companies, weights)[0])
                    p.career.append(school)  # back at school now
            slug = f"{_slugify(first)}-{_slugify(last)}"
            while slug in used_slugs:
                slug = f"{_slugify(first)}-{_slugify(last)}-{rng.randrange(16**6):06x}"
            used_slugs.add(slug)
            p.slug = slug
            self.people.append(p)
            self.adj[pid] = set()

        # Namesakes: different people, same name, same current employer.
        # Only the profile URL tells them apart.
        for _ in range(max(2, n // 400)):
            a, b = rng.sample(self.people, 2)
            if a.career and b.career:
                b.first, b.last = a.first, a.last
                b.career[-1] = a.career[-1]
                b.slug = f"{a.slug}-{rng.randrange(16**6):06x}"

    # -- connections ----------------------------------------------------
    def connect(self, a: int, b: int, when: date | None = None) -> None:
        if a == b:
            return
        key = (min(a, b), max(a, b))
        if key in self.edges:
            return
        if when is None:
            # Nobody connects before the younger person started college.
            start_year = max(self.people[a].grad_year, self.people[b].grad_year) - 4
            earliest = min(date(start_year, 9, 1), AS_OF - timedelta(days=30))
            window = (AS_OF - earliest).days
            days_back = int(self.rng.expovariate(1 / 900))  # skewed recent
            if days_back > window:
                days_back = self.rng.randrange(window)
            when = AS_OF - timedelta(days=days_back)
        self.edges[key] = when
        self.adj[a].add(b)
        self.adj[b].add(a)

    def build_connections(self) -> None:
        rng = self.rng
        by_company: dict[str, list[int]] = {}
        by_cohort: dict[tuple[str, int], list[int]] = {}
        for p in self.people:
            for c in set(p.career):
                by_company.setdefault(c, []).append(p.pid)
            by_cohort.setdefault((p.school, p.grad_year), []).append(p.pid)

        # heavy-tailed target degree (lognormal, median ~30)
        target = {
            p.pid: max(4, min(500, int(math.exp(rng.gauss(3.4, 0.8)) * (0.6 if p.is_student else 1))))
            for p in self.people
        }
        pids = [p.pid for p in self.people]
        popularity = [target[pid] for pid in pids]

        order = pids[:]
        rng.shuffle(order)
        for pid in order:
            person = self.people[pid]
            attempts = 0
            while len(self.adj[pid]) < target[pid] and attempts < target[pid] * 4:
                attempts += 1
                roll = rng.random()
                if roll < 0.5 and person.career:
                    pool = by_company[rng.choice(person.career)]
                elif roll < 0.8:
                    year = person.grad_year + rng.randint(-2, 2)
                    pool = by_cohort.get((person.school, year), [])
                else:
                    pool = None
                if pool:
                    other = rng.choice(pool)
                else:
                    other = rng.choices(pids, popularity)[0]   # preferential
                self.connect(pid, other)


def _messy_row(person: Person, when: date, rng: random.Random) -> dict:
    """Render one connection the way it might appear in someone's export."""
    first, last = person.first, person.last
    nicks = FIRST_NAMES.get(first, [])
    roll = rng.random()
    if nicks and roll < 0.12:
        first = rng.choice(nicks)
    elif roll < 0.15:
        first = first[0] + "."
    if rng.random() < 0.04:
        last = last + rng.choice(_NOTES_CREDENTIALS)
    if rng.random() < 0.04:
        first, last = strip_accents(first), strip_accents(last)
    if rng.random() < 0.03:
        first, last = first.lower(), last.lower()

    company = person.current_company
    variants = COMPANIES.get(company, [company])
    shown_company = rng.choice(variants) if rng.random() < 0.15 else variants[0]
    if rng.random() < 0.02:
        shown_company = ""

    url = "" if rng.random() < 0.07 else f"https://www.linkedin.com/in/{person.slug}"
    email = (
        f"{_slugify(person.first)}.{_slugify(person.last)}@example.com"
        if rng.random() < 0.08 else ""
    )
    return {
        "First Name": first,
        "Last Name": last,
        "URL": url,
        "Email Address": email,
        "Company": shown_company,
        "Position": person.position,
        "Connected On": when.strftime("%d %b %Y"),
    }


def generate(
    out_dir: str | Path,
    *,
    n_people: int = 2500,
    n_members: int = 12,
    seed: int = 42,
) -> dict:
    """Write a simulated dataset to `out_dir` and return its manifest."""
    rng = random.Random(seed)
    world = _World(n_people, rng)
    world.build_connections()
    people = world.people

    # The "club": students and recent grads from the home school.
    candidates = [
        p for p in people
        if p.school == HOME_SCHOOL and AS_OF.year - 1 <= p.grad_year <= AS_OF.year + 3
    ]
    rng.shuffle(candidates)
    seen_ids: set[str] = set()
    members: list[Person] = []
    for p in candidates:
        mid = make_member_id(p.first, p.last)
        if mid in seen_ids:
            continue
        seen_ids.add(mid)
        members.append(p)
        if len(members) == n_members:
            break
    if len(members) < n_members:
        raise ValueError("not enough candidate members; increase n_people")

    alumni = [p for p in people if p.school == HOME_SCHOOL and p.career and not p.is_student]
    for m in members:
        # club members know each other...
        for other in members:
            if other.pid < m.pid and rng.random() < 0.45:
                world.connect(m.pid, other.pid)
        # ...and have met alumni at career fairs and info sessions
        for alum in rng.sample(alumni, k=min(len(alumni), rng.randint(6, 18))):
            world.connect(m.pid, alum.pid)

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "members").mkdir(parents=True)

    truth_rows: list[tuple[str, int, int]] = []
    member_index: dict[str, int] = {}
    for m in members:
        mid = make_member_id(m.first, m.last)
        member_index[mid] = m.pid
        profile = MemberProfile(
            member_id=mid,
            first_name=m.first,
            last_name=m.last,
            url=f"https://www.linkedin.com/in/{m.slug}",
            company=COMPANIES.get(m.current_company, [m.current_company])[0],
            position=m.position,
            school=m.school,
            past_companies=[COMPANIES[c][0] for c in m.career[:-1] if c in COMPANIES],
        )
        neighbors = sorted(world.adj[m.pid], key=lambda o: world.edges[(min(o, m.pid), max(o, m.pid))], reverse=True)
        rows = []
        for idx, other in enumerate(neighbors):
            when = world.edges[(min(other, m.pid), max(other, m.pid))]
            rows.append(_messy_row(people[other], when, rng))
            truth_rows.append((mid, idx, other))
        folder = out / "members" / mid
        folder.mkdir()
        (folder / "profile.json").write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
        (folder / "Connections.csv").write_text(write_connections_csv(rows), encoding="utf-8")

    with open(out / "ground_truth.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["member_id", "row_index", "person_id"])
        writer.writerows(truth_rows)

    manifest = {
        "simulated": True,
        "seed": seed,
        "as_of": AS_OF.isoformat(),
        "n_people": n_people,
        "n_edges_in_world": len(world.edges),
        "members": member_index,
        "rows_written": len(truth_rows),
        "note": "All names are fictional. Company names are used only as labels.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
