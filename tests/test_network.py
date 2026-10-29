import json
import threading
import urllib.error
import urllib.request

import pytest

from pathfinder.evaluate import evaluate
from pathfinder.loader import MemberProfile, write_connections_csv
from pathfinder.network import Network
from pathfinder.server import make_server
from pathfinder.simulate import generate


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("sim")
    manifest = generate(out, n_people=800, n_members=6, seed=7)
    return out, manifest


def test_simulation_is_reproducible(tmp_path):
    a = generate(tmp_path / "a", n_people=300, n_members=3, seed=1)
    b = generate(tmp_path / "b", n_people=300, n_members=3, seed=1)
    assert a["members"] == b["members"]
    for mid in a["members"]:
        assert (tmp_path / "a/members" / mid / "Connections.csv").read_bytes() == \
               (tmp_path / "b/members" / mid / "Connections.csv").read_bytes()


def test_matcher_beats_url_only_baseline(dataset):
    report = evaluate(dataset[0])
    ours, base = report["pathfinder_matcher"], report["url_only_baseline"]
    assert ours["recall"] > base["recall"]
    assert ours["precision"] >= 0.98
    assert ours["recall"] >= 0.95


def test_search_returns_ranked_paths_with_privacy(dataset):
    net = Network.load(dataset[0], persist=False)
    member = next(iter(dataset[1]["members"]))
    company = net.stats()["top_companies"][1]["company"]
    result = net.search(member, company, max_hops=3)
    probs = [r["probability"] for r in result["results"]]
    assert probs and probs == sorted(probs, reverse=True)
    for r in result["results"]:
        assert r["path"][0]["kind"] == "you"
        assert r["path"][-1]["kind"] == "target"
        assert r["hops"] <= 3
        for node in r["path"][2:-1]:
            # deeper non-members are anonymized
            assert node["kind"] in ("member", "hidden")
            if node["kind"] == "hidden":
                assert node["label"].startswith("a contact of ")
        assert r["draft"]["to"] and r["draft"]["body"]


def test_results_are_capped(dataset):
    net = Network.load(dataset[0], persist=False)
    member = next(iter(dataset[1]["members"]))
    company = net.stats()["top_companies"][0]["company"]
    assert len(net.search(member, company, limit=1000)["results"]) <= 25


def _tiny_network(tmp_path):
    net = Network(data_dir=tmp_path)
    rows_a = [{"First Name": "Marcus", "Last Name": "Lee", "URL": "https://www.linkedin.com/in/marcus-lee",
               "Company": "Stripe", "Position": "Engineer", "Connected On": "01 Jan 2020"},
              {"First Name": "Ana", "Last Name": "Ruiz", "URL": "https://www.linkedin.com/in/ana-ruiz",
               "Company": "Midwest State University", "Position": "Student", "Connected On": "01 Jan 2025"}]
    rows_b = [{"First Name": "Bea", "Last Name": "Cole", "URL": "https://www.linkedin.com/in/bea-cole",
               "Company": "Midwest State University", "Position": "Student", "Connected On": "01 Jan 2025"}]
    net.add_member(MemberProfile.from_dict({
        "first_name": "Bea", "last_name": "Cole", "url": "https://www.linkedin.com/in/bea-cole",
        "company": "Midwest State University", "school": "Midwest State University",
        "past_companies": ["Stripe"]}), write_connections_csv(rows_a))
    net.add_member(MemberProfile.from_dict({
        "first_name": "Ana", "last_name": "Ruiz", "url": "https://www.linkedin.com/in/ana-ruiz",
        "company": "Midwest State University", "school": "Midwest State University"}),
        write_connections_csv(rows_b))
    return net


def test_two_hop_intro_and_reasons(tmp_path):
    net = _tiny_network(tmp_path)
    result = net.search("ana-ruiz", "Stripe, Inc.")
    top = result["results"][0]
    assert [n["label"] for n in top["path"]] == ["You", "Bea Cole", "Marcus Lee"]
    assert any("used to work at Stripe" in r for r in top["hop_details"][1]["reasons"])
    assert "Bea" in top["draft"]["body"].split("\n")[0]


def test_remove_member_deletes_their_data(tmp_path):
    net = _tiny_network(tmp_path)
    assert (tmp_path / "members/bea-cole/Connections.csv").exists()
    assert net.remove_member("bea-cole")
    assert not (tmp_path / "members/bea-cole").exists()
    assert net.search("ana-ruiz", "Stripe")["results"] == []
    assert not net.remove_member("../outside")


def test_http_api(tmp_path):
    net = _tiny_network(tmp_path)
    server = make_server(net, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    def call(path, method="GET", body=None):
        req = urllib.request.Request(base + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as res:
                return res.status, json.loads(res.read())
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read())

    try:
        with urllib.request.urlopen(base + "/") as res:
            assert b"Warm Intro Pathfinder" in res.read()
        assert call("/api/stats")[1]["members"] == 2
        status, data = call("/api/search?member=ana-ruiz&company=Stripe")
        assert status == 200 and data["results"][0]["target"]["name"] == "Marcus Lee"
        assert call("/api/search?member=nobody&company=Stripe")[0] == 404
        assert call("/api/members", "POST", {"profile": {"first_name": "X", "last_name": "Y"},
                                             "csv": "not,an,export"})[0] == 422
        assert call("/api/members/bea-cole", "DELETE")[0] == 200
        assert call("/api/stats")[1]["members"] == 1
    finally:
        server.shutdown()
        server.server_close()


def test_generated_files_are_utf8_regardless_of_platform(dataset):
    # Windows' default encoding is cp1252; the generator must not depend on it.
    for path in (dataset[0] / "members").rglob("*"):
        if path.is_file():
            path.read_bytes().decode("utf-8")   # raises if not UTF-8


def test_loads_exports_resaved_as_cp1252(tmp_path):
    # e.g. a Connections.csv opened and saved again in Excel on Windows
    folder = tmp_path / "members" / "rene-dubois"
    folder.mkdir(parents=True)
    profile = {"first_name": "Renée", "last_name": "Dubois", "member_id": "rene-dubois"}
    (folder / "profile.json").write_bytes(json.dumps(profile, ensure_ascii=False).encode("cp1252"))
    rows = [{"First Name": "José", "Last Name": "Núñez", "Company": "Café Labs",
             "Connected On": "01 Jan 2024"}]
    (folder / "Connections.csv").write_bytes(write_connections_csv(rows).encode("cp1252"))

    net = Network.load(tmp_path, persist=False)
    assert net.members()[0]["name"] == "Renée Dubois"
    assert any(p.name == "José Núñez" for p in net.resolution.people)
