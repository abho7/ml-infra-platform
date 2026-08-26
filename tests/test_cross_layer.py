"""
Cross-layer correctness: the composition claim.

Two kinds of test here.

REGRESSIONS pin the composition bugs this project actually found. Each one is a
state no single engine could reach, which is the whole reason the platform
needed its own chaos harness rather than trusting the three suites underneath.

MUTATION TESTS break each invariant deliberately and assert the checker fires,
names the right invariant, and reports where. A correctness harness that has
never been observed to fail is not evidence of anything -- every "0 violations"
on the report is worth exactly as much as this file.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlplat.enginepath  # noqa: F401
from raft.messages import LogEntry

from mlplat.chaos.checker import CrossLayerChecker, ViolationKind
from mlplat.chaos.faults import (
    FaultSchedule, KillNode, KillWorker, Partition, RestartNode,
)
from mlplat.chaos.runner import run
from mlplat.control.plane import ControlPlane, JobState
from mlplat.storage.client import StorageClient
from mlplat.storage.cluster import StorageCluster
from mlplat.storage.statemachine import VectorStateMachine
from mlplat.training.job import JobSpec

DIM = 8


def plane_with_job(steps=30, nodes=5, dim=DIM, seed=7):
    p = ControlPlane(node_ids=[f"n{i + 1}" for i in range(nodes)], dim=dim, seed=seed)
    p.submit(JobSpec(job_id="job", steps=steps, n_classes=dim, publish_every=10))
    ch = CrossLayerChecker(p)
    return p, ch


def kinds(ch):
    return {v["kind"] for v in ch.result.violations}


# ============================================================== REGRESSIONS


def test_regression_same_id_committed_twice_does_not_kill_a_replica():
    """The first real composition bug, pinned.

    A write proposed, timed out unconfirmed, then re-proposed can commit at TWO
    log indices. The watermark cannot help -- both indices are above it, so
    both look new -- and `VectorDB.insert` raises on the duplicate, killing
    apply on every replica at once.

    Neither engine was at fault. Raft committed two entries because it was
    asked to; HNSW refused a duplicate because it should. The defect was in
    treating an unconfirmed write as safe to retry.

    This asserts the state machine survives it, because that is the
    load-bearing half: a state machine over a replicated log must be TOTAL. An
    exception during apply does not reject one bad write, it stops that replica
    applying anything further while its peers continue -- silent divergence
    from a node that still reports itself alive.
    """
    sm = VectorStateMachine(dim=DIM)
    v = np.ones(DIM)
    cmd = {"op": "insert", "id": "dup", "vector": v.tolist()}

    assert sm.apply(LogEntry(term=1, index=1, command=cmd)) is True
    # Same id, DIFFERENT index -- the case the watermark cannot catch.
    assert sm.apply(LogEntry(term=1, index=2, command=cmd)) is False

    assert sm.size == 1
    assert sm.duplicates_suppressed == 1
    assert sm.last_applied_index == 2, "the watermark must still advance"
    # And the machine keeps working afterwards.
    assert sm.apply(LogEntry(term=1, index=3, command={
        "op": "insert", "id": "next", "vector": (v * 2).tolist()})) is True
    assert sm.size == 2


def test_regression_client_does_not_repropose_an_unconfirmed_write():
    """The other half of the fix: never append the same command twice.

    An unconfirmed write returns with an index, so the caller can see it was
    appended and decide -- rather than the client silently appending it again.
    """
    c = StorageCluster([f"n{i + 1}" for i in range(5)], dim=DIM, seed=7)
    c.run_until_leader()
    client = StorageClient(c, commit_timeout=3)

    # Cut the leader off so nothing can commit, then attempt a write.
    leader = c.leader()
    c.partition({leader}, set(c.node_ids) - {leader})
    c.run(3)

    res = client.write("x", np.ones(DIM), {})
    assert res.committed is False
    assert res.reason == "proposed but not committed within the timeout"

    # Exactly ONE entry for this id must exist in the leader's log.
    log = c.replicas[leader].node.log
    appended = [
        i for i in range(1, log.last_index + 1)
        if (log.get(i) or LogEntry(0, 0, {})).command.get("id") == "x"
    ]
    assert len(appended) == 1, (
        f"the client appended {len(appended)} entries for one write; "
        "an unconfirmed proposal was re-proposed"
    )


def test_regression_read_honesty_tests_containment_not_search_recall():
    """The checker's own bug, pinned.

    Invariant 6 once asserted that a k-NN query returned every acked id, and
    the fuzz sweep flagged seed 57. Replaying it showed all acked ids present
    in the serving replica while an approximate query returned only some.

    HNSW is an APPROXIMATE index; a k-NN query is not an enumeration. The
    consistency claim is about the freshness of the state read from, so that is
    what gets asserted, and recall is recorded as an observation.
    """
    p, ch = plane_with_job(steps=20)
    p.run_until_idle()
    pub = set(p.job("job").publisher.published)
    leader = p.cluster.leader()

    # Every acked id is present, so containment must pass...
    ch.check_read_honesty(pub, leader, 999, hits=set())
    assert not ch.result.violations, (
        "read honesty fired despite every acked id being present in the "
        "serving replica -- it is testing recall again"
    )
    # ...and the (deliberately empty) hit set is recorded as an observation.
    assert ch.result.read_recall_samples[-1]["recall"] == 0.0


# ============================================================ MUTATION TESTS


def test_clean_run_reports_no_violations():
    """The control. If this fails the mutations below prove nothing."""
    p, ch = plane_with_job(steps=30)
    while any(m.state is not JobState.DONE for m in p.jobs.values()):
        p.advance()
        ch.check()
    p.cluster.run(120)
    ch.check()
    ch.check_converged()
    assert ch.result.passed, ch.result.violations
    assert ch.result.checks_run > 20


def test_catches_a_replica_holding_more_than_it_applied():
    """Invariant 3: index and watermark disagreeing."""
    p, ch = plane_with_job(steps=20)
    p.run_until_idle()
    ch.check()
    assert ch.result.passed

    # Inject a vector the log never carried.
    victim = p.cluster.live_ids[0]
    p.cluster.replicas[victim].sm._db.insert("ghost", np.ones(DIM), {})
    p.cluster.replicas[victim].sm._ids_in_order.append("ghost")
    p.cluster.replicas[victim].sm._ids_seen.add("ghost")
    ch.check()

    assert not ch.result.passed
    assert ViolationKind.INDEX_WATERMARK in kinds(ch)


def test_catches_a_phantom_embedding():
    """Invariant 4: searchable without a committed log entry."""
    p, ch = plane_with_job(steps=20)
    p.run_until_idle()

    victim = p.cluster.live_ids[0]
    sm = p.cluster.replicas[victim].sm
    sm._db.insert("phantom", np.ones(DIM), {})
    sm._ids_in_order.append("phantom")
    sm._ids_seen.add("phantom")
    # Move the watermark too, so invariant 3 is satisfied and only the phantom
    # check can fire -- otherwise this would not isolate invariant 4.
    sm._last_applied_index += 1
    p.cluster.replicas[victim].observed[sm._last_applied_index] = LogEntry(
        term=1, index=sm._last_applied_index, command={"op": "noop"})
    ch.check()

    assert not ch.result.passed
    assert ViolationKind.PHANTOM in kinds(ch)


def test_catches_a_duplicate_application():
    """Invariant 5: the same embedding stored twice."""
    p, ch = plane_with_job(steps=20)
    p.run_until_idle()

    victim = p.cluster.live_ids[0]
    sm = p.cluster.replicas[victim].sm
    sm._ids_in_order.append(sm._ids_in_order[0])
    ch.check()

    assert not ch.result.passed
    assert ViolationKind.EXACTLY_ONCE in kinds(ch)


def test_catches_replicas_disagreeing_on_their_prefix():
    """Invariant 2: divergence within the common applied prefix."""
    p, ch = plane_with_job(steps=20)
    p.run_until_idle()
    p.cluster.run(60)
    ch.check()
    assert ch.result.passed

    a, b = p.cluster.live_ids[:2]
    order = p.cluster.replicas[b].sm._ids_in_order
    order[0], order[1] = order[1], order[0]   # same set, different order
    ch.check()

    assert not ch.result.passed
    assert ViolationKind.CONVERGENCE in kinds(ch)
    v = next(x for x in ch.result.violations
             if x["kind"] == ViolationKind.CONVERGENCE)
    assert v["position"] == 0


def test_catches_an_acked_write_missing_from_a_new_leader():
    """Invariant 1: durability across a leadership change."""
    p, ch = plane_with_job(steps=20)
    p.run_until_idle()
    ch.check()
    assert ch.result.passed

    # Claim something was acked that the cluster never saw, then force the
    # durability check to run again by advancing the term.
    p.job("job").publisher.published["never-written"] = 99999
    ch._leaders_seen.clear()
    ch.check()

    assert not ch.result.passed
    assert ViolationKind.DURABILITY in kinds(ch)


def test_catches_a_read_served_from_stale_state():
    """Invariant 6: served from a replica missing an acked write."""
    p, ch = plane_with_job(steps=20)
    p.run_until_idle()
    pub = set(p.job("job").publisher.published)

    # A replica that was reset and never caught up.
    stale = p.cluster.live_ids[-1]
    p.cluster.replicas[stale].sm.reset()
    ch.check_read_honesty(pub, stale, 42)

    assert not ch.result.passed
    assert ViolationKind.READ_HONESTY in kinds(ch)


def test_checker_does_not_fire_on_healthy_runs_across_sizes():
    """A checker that fires on correct systems is as useless as one that never
    fires."""
    for nodes in (3, 5, 7):
        p, ch = plane_with_job(steps=20, nodes=nodes, seed=nodes * 11)
        while any(m.state is not JobState.DONE for m in p.jobs.values()):
            p.advance()
            ch.check()
        p.cluster.run(120)
        ch.check()
        ch.check_converged()
        assert ch.result.passed, f"{nodes} nodes: {ch.result.violations}"


# ============================================================== SCENARIOS


@pytest.mark.slow
@pytest.mark.parametrize("scenario_name", [
    "worker_dies_during_partition", "node_dies_mid_write", "cascade",
    "minority_side_isolation", "rejoin_after_long_absence", "everything_at_once",
])
def test_named_scenarios_hold(scenario_name):
    from mlplat.chaos import scenarios
    r = getattr(scenarios, scenario_name)()
    assert r.passed, r.check["violations"]
    assert r.published == r.queryable, (
        f"{r.published - r.queryable} embeddings were acked but are not "
        "queryable afterwards"
    )
    assert r.converged, "replicas failed to converge after the faults healed"


@pytest.mark.slow
def test_cross_layer_faults_actually_overlap():
    """Guards the headline claim from being vacuous.

    A scenario that says it tests simultaneous cross-layer failure must be able
    to show the failures were in fact simultaneous AND in different layers.
    """
    from mlplat.chaos import scenarios
    r = scenarios.worker_dies_during_partition()
    assert r.cross_layer_overlaps >= 1
    r2 = scenarios.everything_at_once()
    assert r2.cross_layer_overlaps >= 2


@pytest.mark.slow
def test_fuzz_sweep_is_clean():
    from mlplat.chaos.fuzz import sweep
    s = sweep(12, quiet=True)
    assert s["failed"] == 0, s["failing_seeds"]
    assert s["published_not_queryable"] == 0
    assert s["unconverged"] == 0
    assert s["total_published"] > 0
