"""
Named cross-layer scenarios.

Each one exists because it is a state that CANNOT occur inside any single
engine. raft-chaos-testing can partition a cluster but has no notion of a
training worker; the training framework can kill a worker but has no notion of
quorum. Everything below needs both.

The hand-built scenarios are good report material and poor bug hunters -- they
only probe interleavings somebody thought of. `fuzz.py` exists for the same
reason it does in the other two projects.
"""

from __future__ import annotations

from mlplat.chaos.faults import (
    DelayLink, FaultSchedule, KillNode, KillWorker, Partition, RestartNode,
)
from mlplat.chaos.runner import ScenarioResult, run


def clean(seed: int = 7) -> ScenarioResult:
    return run(
        name="No faults",
        slug="clean",
        description=(
            "A training job publishes embeddings into a healthy five-node "
            "cluster. The control: establishes that the composition works and "
            "that the checker is silent on a healthy run, without which none "
            "of the scenarios below prove anything."
        ),
        fault_summary="none",
        seed=seed,
    )


def worker_dies_during_partition(seed: int = 7) -> ScenarioResult:
    return run(
        name="Worker dies while storage is partitioned",
        slug="worker-dies-during-partition",
        description=(
            "The headline cross-layer case. The storage cluster splits 2/3 at "
            "round 15, and a training worker is killed at round 20 while the "
            "split is still active. Two layers degraded at once: the job must "
            "keep training on fewer workers while the minority side cannot "
            "commit. Neither engine can produce this state alone."
        ),
        fault_summary="Partition [n1,n2]|[n3,n4,n5] rounds 15-45; KillWorker at 20",
        schedule=FaultSchedule([
            Partition(at=15, until=45, groups=[["n1", "n2"], ["n3", "n4", "n5"]]),
            KillWorker(at=20, job="job"),
        ]),
        seed=seed,
    )


def node_dies_mid_write(seed: int = 11) -> ScenarioResult:
    return run(
        name="Storage node dies mid-write",
        slug="node-dies-mid-write",
        description=(
            "A follower dies at round 10 and the leader at round 25, both "
            "while a job is actively publishing. Acknowledged embeddings must "
            "survive the failover. An un-acknowledged in-flight write "
            "vanishing is correct -- the analogue of an uncommitted Raft entry "
            "-- and is counted, not failed on."
        ),
        fault_summary="KillNode n2@10, KillNode on the leader@25",
        schedule=FaultSchedule([
            KillNode(at=10, node="n2"),
            KillNode(at=25, node="n1"),
        ]),
        seed=seed,
    )


def slow_link_causes_ambiguous_writes(seed: int = 13) -> ScenarioResult:
    return run(
        name="Slow link during publication",
        slug="slow-link",
        description=(
            "A link is delayed rather than cut, so commits still happen but "
            "slowly. This is the condition that produces the genuinely "
            "ambiguous write: proposed, timed out at the client, and possibly "
            "committed later anyway. Retrying such a write is what caused this "
            "project's first real composition bug -- see Findings."
        ),
        fault_summary="DelayLink n1->n3 and n1->n4 by 8 ticks, rounds 12-40",
        schedule=FaultSchedule([
            DelayLink(at=12, until=40, src="n1", dst="n3", ticks=8),
            DelayLink(at=12, until=40, src="n1", dst="n4", ticks=8),
        ]),
        seed=seed,
    )


def cascade(seed: int = 17) -> ScenarioResult:
    return run(
        name="Cascading failures across both layers",
        slug="cascade",
        description=(
            "Five faults in sequence rather than in isolation: a node dies, a "
            "worker dies, the cluster partitions, a second worker dies during "
            "the split, and the first node rejoins afterwards. Recovery from "
            "each must not depend on the others having finished."
        ),
        fault_summary=(
            "KillNode n5@8; KillWorker@14; Partition rounds 20-38; "
            "KillWorker@26; RestartNode n5@42"
        ),
        schedule=FaultSchedule([
            KillNode(at=8, node="n5"),
            KillWorker(at=14, job="job"),
            Partition(at=20, until=38, groups=[["n1"], ["n2", "n3", "n4"], ["n5"]]),
            KillWorker(at=26, job="job"),
            RestartNode(at=42, node="n5"),
        ]),
        n_workers=5,
        seed=seed,
    )


def minority_side_isolation(seed: int = 19) -> ScenarioResult:
    return run(
        name="Leader isolated on the minority side",
        slug="leader-isolated",
        description=(
            "The leader is cut off alone while publication continues. It still "
            "believes it is leader and still has a plausible commit index. "
            "Linearizable reads against it must be refused rather than served "
            "stale, and the majority side must elect a new leader and carry "
            "on. This is precisely what ReadIndex was implemented for."
        ),
        fault_summary="Partition [n1]|[n2..n5] rounds 18-50",
        schedule=FaultSchedule([
            Partition(at=18, until=50, groups=[["n1"], ["n2", "n3", "n4", "n5"]]),
        ]),
        seed=seed,
    )


def rejoin_after_long_absence(seed: int = 23) -> ScenarioResult:
    return run(
        name="Replica rejoins after missing many writes",
        slug="rejoin",
        description=(
            "A replica is killed early, misses most of a job's publications, "
            "and rejoins near the end. With no snapshot support in the engine, "
            "catch-up is a full log replay -- which must reconstruct an index "
            "byte-identical to its peers'."
        ),
        fault_summary="KillNode n4@6, RestartNode n4@45",
        schedule=FaultSchedule([
            KillNode(at=6, node="n4"),
            RestartNode(at=45, node="n4"),
        ]),
        seed=seed,
    )


def everything_at_once(seed: int = 29) -> ScenarioResult:
    return run(
        name="Simultaneous faults in both layers",
        slug="simultaneous",
        description=(
            "A node death, a worker death, a partition and a slow link, all "
            "with overlapping windows. Faults compose because they are "
            "independent objects consulted per round, so this needed no "
            "special handling anywhere in the platform."
        ),
        fault_summary=(
            "KillNode n5@10; KillWorker@10; Partition rounds 10-40; "
            "DelayLink n1->n2 rounds 10-40"
        ),
        schedule=FaultSchedule([
            KillNode(at=10, node="n5"),
            KillWorker(at=10, job="job"),
            Partition(at=10, until=40, groups=[["n1", "n2"], ["n3", "n4"], ["n5"]]),
            DelayLink(at=10, until=40, src="n1", dst="n2", ticks=6),
        ]),
        n_workers=5,
        seed=seed,
    )


SCENARIOS = [
    clean,
    worker_dies_during_partition,
    node_dies_mid_write,
    slow_link_causes_ambiguous_writes,
    cascade,
    minority_side_isolation,
    rejoin_after_long_absence,
    everything_at_once,
]


def run_all() -> list[ScenarioResult]:
    return [s() for s in SCENARIOS]
