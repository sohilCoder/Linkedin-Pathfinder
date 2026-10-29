import math
import random

import pytest

from pathfinder.search import (
    brute_force_best_paths,
    hop_limited_dijkstra,
    path_cost,
    probability_to_cost,
)


def random_graph(rng, n, p):
    adj = {i: [] for i in range(n)}
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < p:
                w = probability_to_cost(rng.uniform(0.05, 0.95))
                adj[a].append((b, w))
                adj[b].append((a, w))
    return adj


@pytest.mark.parametrize("seed", range(40))
def test_matches_brute_force_on_random_graphs(seed):
    rng = random.Random(seed)
    adj = random_graph(rng, n=rng.randint(4, 11), p=rng.uniform(0.15, 0.6))
    for max_hops in (1, 2, 3, 4):
        got = hop_limited_dijkstra(adj, 0, max_hops)
        want = brute_force_best_paths(adj, 0, max_hops)
        assert set(got) == set(want)
        for node, (cost, path) in got.items():
            assert math.isclose(cost, want[node], abs_tol=1e-9)
            assert path[0] == 0 and path[-1] == node
            assert len(path) - 1 <= max_hops
            assert len(set(path)) == len(path)          # simple path
            assert math.isclose(path_cost(adj, path), cost, abs_tol=1e-9)


def test_prefers_short_weaker_path_when_long_one_exceeds_hop_limit():
    # 0 -> 1 -> 2 -> 3 -> T is very strong; 0 -> 4 -> T is weaker but 2 hops.
    strong, weak = probability_to_cost(0.95), probability_to_cost(0.4)
    edges = [(0, 1, strong), (1, 2, strong), (2, 3, strong), (3, "T", strong),
             (0, 4, weak), (4, "T", weak)]
    adj = {}
    for a, b, w in edges:
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))

    unlimited = hop_limited_dijkstra(adj, 0, 10)["T"]
    assert unlimited[1] == [0, 1, 2, 3, "T"]
    limited = hop_limited_dijkstra(adj, 0, 3)["T"]
    assert limited[1] == [0, 4, "T"]


def test_unreachable_and_zero_hops():
    adj = {0: [(1, 0.1)], 1: [(0, 0.1)], 2: []}
    assert set(hop_limited_dijkstra(adj, 0, 0)) == {0}
    assert 2 not in hop_limited_dijkstra(adj, 0, 5)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        probability_to_cost(0)
    with pytest.raises(ValueError):
        hop_limited_dijkstra({0: [(1, -1.0)]}, 0, 2)
