from pathfinder.matching import Mention, pairwise_scores, resolve


def m(n, first, last, company="Acme", url=""):
    return Mention(("row", "test", n), first, last, url, company, "Engineer")


def same(res, a, b):
    return res.person_of[("row", "test", a)] == res.person_of[("row", "test", b)]


def test_same_url_always_merges_even_with_unknown_nickname():
    res = resolve([
        m(1, "Elizabeth", "Van Dyke", url="linkedin.com/in/evd"),
        m(2, "Libby", "Van Dyke, PMP", url="https://www.linkedin.com/in/evd/"),
    ])
    assert same(res, 1, 2)


def test_nickname_without_url_joins_known_identity():
    res = resolve([
        m(1, "Jonathan", "Smith", "Meta Platforms, Inc.", url="linkedin.com/in/js"),
        m(2, "Jon", "Smith", "Facebook"),
    ])
    assert same(res, 1, 2)


def test_namesakes_with_different_urls_stay_apart():
    res = resolve([
        m(1, "Alex", "Kim", url="linkedin.com/in/alex-kim-1"),
        m(2, "Alex", "Kim", url="linkedin.com/in/alex-kim-2"),
    ])
    assert not same(res, 1, 2)


def test_ambiguous_mention_is_not_guessed():
    # A URL-less "Alex Kim" at Acme could be either namesake: leave it alone.
    res = resolve([
        m(1, "Alex", "Kim", url="linkedin.com/in/alex-kim-1"),
        m(2, "Alex", "Kim", url="linkedin.com/in/alex-kim-2"),
        m(3, "Alex", "Kim"),
    ])
    assert not same(res, 3, 1) and not same(res, 3, 2)
    assert res.stats["ambiguous_left_unmerged"] == 1


def test_initials_do_not_chain_different_people():
    res = resolve([m(1, "John", "Doe"), m(2, "J.", "Doe"), m(3, "Jane", "Doe")])
    assert not same(res, 1, 3)


def test_different_company_is_not_merged_without_url():
    res = resolve([m(1, "Priya", "Shah", "Google"), m(2, "Priya", "Shah", "Stripe")])
    assert not same(res, 1, 2)


def test_url_backed_cluster_with_blank_company_still_matches():
    # The URL'd mention has no company, but another mention of the same URL
    # does, so the cluster's company is known and the URL-less row can join.
    res = resolve([
        m(1, "Tim", "Lee", "", url="linkedin.com/in/tl"),
        m(2, "Timothy", "Lee", "Acme", url="linkedin.com/in/tl"),
        m(3, "Timothy", "Lee", "Acme"),
    ])
    assert same(res, 1, 3)


def test_url_only_baseline_misses_fuzzy_matches():
    mentions = [m(1, "Jonathan", "Smith", url="linkedin.com/in/js"), m(2, "Jon", "Smith")]
    assert not same(resolve(mentions, fuzzy=False), 1, 2)


def test_pairwise_scores():
    truth = {"a": 1, "b": 1, "c": 2, "d": 2}
    perfect = pairwise_scores({"a": 0, "b": 0, "c": 1, "d": 1}, truth)
    assert perfect["precision"] == perfect["recall"] == 1.0
    lumped = pairwise_scores({"a": 0, "b": 0, "c": 0, "d": 0}, truth)
    assert lumped["recall"] == 1.0 and lumped["precision"] == 2 / 6
    split = pairwise_scores({"a": 0, "b": 1, "c": 2, "d": 3}, truth)
    assert split["recall"] == 0.0 and split["precision"] == 1.0
