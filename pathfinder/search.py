"""Hop-limited shortest paths (Dijkstra over (node, hops) states).

Why not plain Dijkstra? Intro chains longer than ~3 people almost never work,
so we want the *strongest path that uses at most K hops*. Plain Dijkstra can
return a long chain of strong ties when a slightly weaker 2-hop path exists,
and then has no way to offer the short one.

The fix is to search over states (node, hops_used). A state is only expanded
while hops_used < K. One pruning rule keeps it fast: if we have already
settled `node` with *fewer or equal* hops, any later state for that node
also has a higher-or-equal cost (Dijkstra pops in cost order) and more hops,
so it is dominated and can be skipped.

Edge costs are -log(probability the intro happens). Adding costs along a
path multiplies the probabilities, so "shortest" means "most likely to work".
"""

from __future__ import annotations

import heapq
import math
from typing import Hashable, Iterable, Mapping, Sequence

Node = Hashable
Adjacency = Mapping[Node, Sequence[tuple[Node, float]]]


def probability_to_cost(p: float) -> float:
    if not 0.0 < p <= 1.0:
        raise ValueError(f"probability must be in (0, 1], got {p}")
    return -math.log(p)


def cost_to_probability(cost: float) -> float:
    return math.exp(-cost)


def hop_limited_dijkstra(
    adj: Adjacency, source: Node, max_hops: int
) -> dict[Node, tuple[float, list[Node]]]:
    """Best path from `source` to every node reachable in <= max_hops edges.

    Returns {node: (total_cost, [source, ..., node])}. Costs must be >= 0.
    """
    if max_hops < 0:
        raise ValueError("max_hops must be >= 0")
    best: dict[Node, tuple[float, list[Node]]] = {source: (0.0, [source])}
    dist: dict[tuple[Node, int], float] = {(source, 0): 0.0}
    parent: dict[tuple[Node, int], tuple[Node, int]] = {}
    min_settled_hops: dict[Node, int] = {}
    heap: list[tuple[float, int, int, Node]] = [(0.0, 0, 0, source)]
    tiebreak = 1   # keeps the heap from ever comparing node objects

    while heap:
        cost, hops, _, node = heapq.heappop(heap)
        if cost > dist.get((node, hops), math.inf):
            continue   # stale entry
        if min_settled_hops.get(node, math.inf) <= hops:
            continue   # dominated: cheaper-or-equal with fewer-or-equal hops exists
        min_settled_hops[node] = hops

        if node not in best:
            best[node] = (cost, _rebuild(parent, node, hops))
        if hops == max_hops:
            continue
        for nbr, weight in adj.get(node, ()):
            if weight < 0:
                raise ValueError("edge costs must be non-negative")
            state = (nbr, hops + 1)
            new_cost = cost + weight
            if new_cost < dist.get(state, math.inf):
                dist[state] = new_cost
                parent[state] = (node, hops)
                heapq.heappush(heap, (new_cost, hops + 1, tiebreak, nbr))
                tiebreak += 1
    return best


def _rebuild(parent: dict, node: Node, hops: int) -> list[Node]:
    path = [node]
    state = (node, hops)
    while state in parent:
        state = parent[state]
        path.append(state[0])
    path.reverse()
    return path


def brute_force_best_paths(
    adj: Adjacency, source: Node, max_hops: int
) -> dict[Node, float]:
    """Reference implementation for tests: enumerate every simple path."""
    best: dict[Node, float] = {source: 0.0}

    def walk(node: Node, cost: float, hops: int, seen: set) -> None:
        if hops == max_hops:
            return
        for nbr, weight in adj.get(node, ()):
            if nbr in seen:
                continue
            total = cost + weight
            if total < best.get(nbr, math.inf):
                best[nbr] = total
            seen.add(nbr)
            walk(nbr, total, hops + 1, seen)
            seen.remove(nbr)

    walk(source, 0.0, 0, {source})
    return best


def path_cost(adj: Adjacency, path: Iterable[Node]) -> float:
    path = list(path)
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += min(w for n, w in adj[a] if n == b)
    return total
