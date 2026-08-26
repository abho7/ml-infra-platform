"""
Storage layer: a Raft-replicated HNSW index.

These tests are about the SEAM, not about either engine. Raft's own safety
properties are proven by its 23 tests and HNSW's recall by its 28; what is
unproven until here is that wiring them together preserves both.

Two hazards get pinned rather than merely avoided, because a test that only
shows the fixed version working does not demonstrate that the fix was
load-bearing:

  * re-applying a committed prefix (which Raft requires) against an index that
    refuses duplicates (which HNSW does)
  * replicas built with different RNG seeds, which produce different graphs
    from the identical committed log
"""

from __future__ import annotations

import numpy as np
import pytest

from mlplat.storage.client import StorageClient
from mlplat.storage.cluster import StorageCluster
from mlplat.storage.readindex import ReadMode, ReadRefused
from mlplat.storage.statemachine import VectorStateMachine

import mlplat.enginepath  # noqa: F401
from raft.messages import LogEntry

DIM = 8


def make_cluster(n=5, *, seed=7, dim=DIM):
    c = StorageCluster([f"n{i + 1}" for i in range(n)], dim=dim, seed=seed)
    assert c.run_until_leader() is not None, "cluster never elected a leader"
    return c


def fill(client, n, *, rng_seed=1, dim=DIM):
    rng = np.random.default_rng(rng_seed)
    vecs = {}
    for i in range(n):
        v = rng.normal(size=dim)
        vecs[f"e{i}"] = v
        r = client.write(f"e{i}", v, {"i": i})
        assert r.committed, f"write e{i} was not committed: {r.reason}"
    return vecs


# ----------------------------------------------------------------- basics


def test_cluster_elects_a_leader_and_commits_writes():
    c = make_cluster()
    cl = StorageClient(c)
    fill(cl, 10)

    assert len(cl.acked) == 10
    leader = c.leader()
    assert leader is not None
    assert c.replicas[leader].sm.size == 10


def test_every_replica_can_answer_the_same_query():
    """The point of replication: any in-sync replica serves the same answer."""
    c = make_cluster()
    cl = StorageClient(c)
    vecs = fill(cl, 20)
    c.run(40)  # let followers catch up

    q = vecs["e7"]
    answers = {nid: [h["id"] for h in c.search_on(nid, q, 5)] for nid in c.live_ids}
    assert len({tuple(v) for v in answers.values()}) == 1, answers
    assert answers["n1"][0] == "e7", "nearest neighbour of a stored vector is not itself"


# --------------------------------------------------- hazard 1: idempotency


def test_state_machine_is_idempotent_by_index():
    """Raft re-delivers a restarted node's whole prefix. HNSW refuses
    duplicates. The watermark is the adapter between the two."""
    sm = VectorStateMachine(dim=DIM)
    v = np.ones(DIM)
    e = LogEntry(term=1, index=1, command={"op": "insert", "id": "a", "vector": v.tolist()})

    assert sm.apply(e) is True
    assert sm.apply(e) is False, "re-applying the same index should be a no-op"
    assert sm.apply(e) is False
    assert sm.size == 1
    assert sm.last_applied_index == 1


def test_without_the_watermark_a_replay_would_raise():
    """Pins the hazard: proves the watermark is load-bearing, not decoration.

    Bypasses `apply` and calls the underlying VectorDB the way a naive wiring
    would. If this ever stops raising, the engine changed and the watermark
    above may no longer be necessary -- which is worth being told about.
    """
    sm = VectorStateMachine(dim=DIM)
    v = np.ones(DIM)
    sm._db.insert("a", v, {})
    with pytest.raises(ValueError, match="already exists"):
        sm._db.insert("a", v, {})


def test_restart_replays_the_log_without_duplicating():
    """The real path: kill a follower, restart it, let it catch up."""
    c = make_cluster()
    cl = StorageClient(c)
    fill(cl, 15)
    c.run(40)

    victim = next(n for n in c.live_ids if n != c.leader())
    before = c.replicas[victim].sm.digest()
    assert before[1] == 15

    c.kill(victim)
    c.run(20)
    c.restart(victim)

    # Immediately after restart the replica must be empty, not stale.
    assert c.replicas[victim].sm.size == 0
    assert c.replicas[victim].sm.last_applied_index == 0

    c.run(200)
    after = c.replicas[victim].sm.digest()
    assert after == before, f"replica did not replay to the same state: {after} != {before}"


# ------------------------------------------- hazard 2: index/watermark unity


def test_reset_clears_index_and_watermark_together():
    sm = VectorStateMachine(dim=DIM)
    for i in range(5):
        sm.apply(LogEntry(term=1, index=i + 1,
                          command={"op": "insert", "id": f"x{i}",
                                   "vector": np.full(DIM, float(i)).tolist()}))
    assert (sm.size, sm.last_applied_index) == (5, 5)

    sm.reset()
    assert (sm.size, sm.last_applied_index) == (0, 0), (
        "reset left the index and the watermark disagreeing, which is the "
        "silent-data-loss hazard"
    )
    # And it can be refilled from index 1 without a duplicate error.
    assert sm.apply(LogEntry(term=1, index=1,
                             command={"op": "insert", "id": "x0",
                                      "vector": np.zeros(DIM).tolist()})) is True


def test_index_matches_the_claimed_applied_prefix():
    """Invariant 3, directly: a replica's content equals what it claims."""
    c = make_cluster()
    cl = StorageClient(c)
    fill(cl, 12)
    c.run(40)

    for nid, r in c.replicas.items():
        inserts = sum(
            1 for i in sorted(r.observed)
            if i <= r.sm.last_applied_index
            and r.observed[i].command.get("op") == "insert"
        )
        assert r.sm.size == inserts, (
            f"{nid} holds {r.sm.size} vectors but claims to have applied "
            f"{inserts} inserts up to index {r.sm.last_applied_index}"
        )


# ------------------------------------------------- hazard 3: seed determinism


def test_replicas_share_a_seed_and_converge_bitwise():
    c = make_cluster()
    cl = StorageClient(c)
    fill(cl, 25)
    c.run(60)  # quiesce

    digests = c.digests()
    assert len(set(digests.values())) == 1, (
        f"replicas diverged despite an identical committed log: {digests}"
    )


def test_seed_changes_routing_but_not_answers():
    """What the shared seed actually buys, measured.

    This test originally asserted that differently-seeded replicas return
    DIFFERENT results, on the assumption that a shared seed was preventing a
    real divergence hazard. It failed, and the assumption was wrong. The
    measured truth is more specific:

      * layer 0 is identical regardless of seed -- it holds every node and its
        edges come from the distance-based diversity heuristic, no RNG, so it
        is fully determined by insertion order (which Raft already makes
        identical across replicas)
      * the upper layers do differ, since the per-insert level draw is the only
        use of the RNG
      * search results agree anyway, because the upper layers only pick where
        the layer-0 search begins

    So the shared seed is defensive, not load-bearing at this scale. Asserting
    it prevents a divergence that does not occur would be a fabricated claim;
    this asserts what was actually observed.
    """
    rng = np.random.default_rng(5)
    n = 300
    vecs = rng.normal(size=(n, DIM))

    def build(seed):
        sm = VectorStateMachine(dim=DIM, seed=seed)
        for i, v in enumerate(vecs):
            sm.apply(LogEntry(term=1, index=i + 1,
                              command={"op": "insert", "id": f"v{i}",
                                       "vector": v.tolist()}))
        return sm

    a, b = build(1), build(999)
    assert a.ids_in_order == b.ids_in_order

    ia, ib = a._db._index, b._db._index
    assert ia.layers[0] == ib.layers[0], (
        "layer 0 differed between seeds; it is supposed to be RNG-free and "
        "determined entirely by insertion order"
    )
    assert (ia.max_layer, ia.entry_point) != (ib.max_layer, ib.entry_point), (
        "the seed changed nothing at all, so it cannot be said to control "
        "graph construction"
    )

    for q in rng.normal(size=(25, DIM)):
        ra = [h["id"] for h in a.search(q, 10, ef_search=4)]
        rb = [h["id"] for h in b.search(q, 10, ef_search=4)]
        assert ra == rb, (
            "differently-seeded indexes disagreed, so the shared seed IS "
            "load-bearing at this scale and the comment in statemachine.py "
            "calling it defensive is now wrong"
        )


# ------------------------------------------------------- prefix vs converged


def test_replicas_always_agree_on_their_common_prefix():
    """Instantaneously true, unlike full convergence.

    Followers legitimately lag the leader by an entry or two: commit_index
    propagates on the next heartbeat. So the always-true invariant is prefix
    agreement, and full digest equality is only guaranteed after quiescence.
    Asserting the stronger one continuously would fail on correct behaviour.
    """
    c = make_cluster()
    cl = StorageClient(c)
    rng = np.random.default_rng(3)

    for i in range(20):
        cl.write(f"e{i}", rng.normal(size=DIM), {"i": i})
        orders = [r.sm.ids_in_order for r in c.replicas.values() if r.alive]
        m = min(len(o) for o in orders)
        assert len({tuple(o[:m]) for o in orders}) == 1, (
            f"replicas disagree on their common prefix of length {m}"
        )


def test_followers_do_lag_before_quiescing():
    """Guards the test above from becoming vacuous.

    If followers never lagged, prefix agreement and full convergence would be
    the same statement and the distinction would be untested.
    """
    c = make_cluster()
    cl = StorageClient(c)
    fill(cl, 15)
    applied = {nid: r.sm.last_applied_index for nid, r in c.replicas.items()}
    assert len(set(applied.values())) > 1, (
        f"no follower lagged the leader, so this scenario cannot distinguish "
        f"prefix agreement from convergence: {applied}"
    )
    c.run(60)
    assert len({r.sm.digest() for r in c.replicas.values()}) == 1


# ------------------------------------------------------------------ reads


def test_linearizable_read_sees_every_acked_write():
    c = make_cluster()
    cl = StorageClient(c)
    vecs = fill(cl, 18)

    for probe in ("e0", "e9", "e17"):
        res = cl.search(vecs[probe], k=1)
        assert res.hits[0]["id"] == probe
        assert res.read_index is not None and res.read_index >= 18


def test_linearizable_read_is_refused_on_the_minority_side():
    """The whole reason ReadIndex exists.

    A partitioned ex-leader still has role == LEADER and a plausible
    commit_index. Without the quorum confirmation round it would happily serve
    stale data and look correct.
    """
    c = make_cluster()
    cl = StorageClient(c)
    fill(cl, 10)
    c.run(30)

    leader = c.leader()
    minority = {leader}
    majority = set(c.node_ids) - minority
    c.partition(minority, majority)
    c.run(5)

    # A read aimed at the isolated node must fail rather than lie.
    with pytest.raises(ReadRefused):
        stranded = StorageClient(c)
        ticket = stranded.reader.begin()
        stranded.reader.confirm(ticket, max_ticks=80)


def test_stale_read_still_succeeds_during_a_partition():
    """The other half of the contract: stale reads keep working, and are
    allowed to be old. That is the mode's promise, not a bug."""
    c = make_cluster()
    cl = StorageClient(c)
    vecs = fill(cl, 10)
    c.run(30)

    leader = c.leader()
    isolated = {leader}
    c.partition(isolated, set(c.node_ids) - isolated)
    c.run(5)

    res = cl.search(vecs["e3"], k=1, mode=ReadMode.STALE, node_id=leader)
    assert res.hits[0]["id"] == "e3"
    assert res.mode == "stale"


def test_majority_side_keeps_serving_linearizable_reads():
    c = make_cluster()
    cl = StorageClient(c)
    vecs = fill(cl, 10)
    c.run(30)

    old_leader = c.leader()
    minority = {old_leader}
    c.partition(minority, set(c.node_ids) - minority)
    c.run(120)  # majority side elects a new leader

    new_leader = c.leader()
    assert new_leader is not None and new_leader != old_leader

    res = cl.search(vecs["e5"], k=1)
    assert res.hits[0]["id"] == "e5"


# ------------------------------------------------------------- durability


def test_acked_writes_survive_leader_death():
    c = make_cluster()
    cl = StorageClient(c)
    vecs = fill(cl, 12)
    c.run(40)

    leader = c.leader()
    c.kill(leader)
    assert c.run_until_leader(max_ticks=300) is not None
    c.run(120)

    survivor = c.leader()
    for eid in cl.acked:
        assert c.replicas[survivor].sm.contains(eid), (
            f"acked write {eid} is missing from the new leader after failover"
        )


def test_rejoining_replica_catches_up_to_the_others():
    c = make_cluster()
    cl = StorageClient(c)
    fill(cl, 10)
    c.run(30)

    victim = next(n for n in c.live_ids if n != c.leader())
    c.kill(victim)

    for i in range(10, 20):
        r = cl.write(f"e{i}", np.random.default_rng(i).normal(size=DIM), {"i": i})
        assert r.committed

    c.restart(victim)
    c.run(300)

    assert c.replicas[victim].sm.size == 20
    assert c.replicas[victim].sm.digest() == c.replicas[c.leader()].sm.digest()
