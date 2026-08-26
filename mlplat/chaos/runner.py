"""
Runs one cross-layer scenario to completion, checking correctness every round.

Shared by the named scenarios and the randomized sweep so both drive the
platform through exactly the same path. If they had separate loops, a scenario
could pass and its fuzzed equivalent fail for reasons having nothing to do with
the faults, and the report would be comparing two different systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mlplat.chaos.checker import CrossLayerChecker
from mlplat.chaos.faults import FaultSchedule
from mlplat.control.plane import ControlPlane, JobState
from mlplat.observe.events import EventLog
from mlplat.storage.readindex import ReadMode, ReadRefused
from mlplat.training.job import JobSpec


@dataclass
class ScenarioResult:
    name: str
    slug: str
    description: str
    fault_summary: str
    seed: int
    nodes: list[str]
    rounds: int
    passed: bool
    check: dict
    events: list[dict] = field(default_factory=list)
    fault_windows: list[dict] = field(default_factory=list)
    cross_layer_overlaps: int = 0
    jobs: list[dict] = field(default_factory=list)
    health_timeline: list[dict] = field(default_factory=list)
    published: int = 0
    queryable: int = 0
    lost_uncommitted: int = 0
    final_leader: str | None = None
    converged: bool = False

    def to_json(self) -> dict:
        return dict(self.__dict__)


def run(
    *,
    name: str,
    slug: str,
    description: str,
    schedule: FaultSchedule | None = None,
    fault_summary: str = "none",
    n_nodes: int = 5,
    n_workers: int = 4,
    steps: int = 60,
    dim: int = 8,
    seed: int = 7,
    publish_every: int = 12,
    settle_rounds: int = 250,
) -> ScenarioResult:
    schedule = schedule or FaultSchedule()
    events = EventLog()
    plane = ControlPlane(
        node_ids=[f"n{i + 1}" for i in range(n_nodes)],
        dim=dim, seed=seed, events=events,
    )
    checker = CrossLayerChecker(plane, events=events)

    spec = JobSpec(
        job_id="job", n_workers=n_workers, steps=steps,
        n_classes=dim, publish_every=publish_every, seed=seed + 100,
    )
    plane.submit(spec)

    health_timeline: list[dict] = []
    rnd = 0
    while any(
        m.state not in (JobState.DONE, JobState.FAILED) for m in plane.jobs.values()
    ):
        schedule.apply(plane, rnd)
        plane.advance()
        checker.check()

        if rnd % 5 == 0:
            h = plane.state()["health"]
            health_timeline.append({
                "round": rnd, "tick": plane.cluster.tick,
                "status": h["status"], "live": h["live_nodes"],
                "quorum": h["has_quorum"], "leader": h["leader"],
            })

        rnd += 1
        if rnd > steps * 40:
            break  # a wedged scenario is a result, not a hang

    # Let everything settle: heal, revive, and give the cluster time to
    # converge. The post-fault steady state is where convergence and durability
    # become assertable, and a scenario that ends mid-election proves nothing.
    plane.heal()
    for nid in plane.cluster.dead_ids:
        plane.restart_node(nid)
    plane.cluster.run(settle_rounds)
    checker.check()
    checker.check_converged()

    managed = plane.job("job")
    managed.publisher.resolve_pending()

    leader = plane.cluster.leader()
    published = set(managed.publisher.published)
    queryable = (
        {e for e in published if plane.cluster.replicas[leader].sm.contains(e)}
        if leader else set()
    )

    # Read honesty, on the real read path rather than by inspection.
    if leader and published:
        try:
            probe = plane.cluster.replicas[leader].sm._db._index.vectors[
                plane.cluster.replicas[leader].sm._db._id_map[sorted(published)[0]]
            ]
            res = plane.client.search(np.asarray(probe), k=len(published) + 5,
                                      mode=ReadMode.LINEARIZABLE)
            checker.check_read_honesty(
                published, res.served_by, res.read_index or 0,
                hits={h["id"] for h in res.hits},
            )
        except ReadRefused:
            checker.note_minority_refusal()

    checker.note_uncommitted_loss(len(managed.publisher.rejected))

    digests = plane.cluster.digests()
    return ScenarioResult(
        name=name, slug=slug, description=description,
        fault_summary=fault_summary, seed=seed,
        nodes=plane.cluster.node_ids, rounds=rnd,
        passed=checker.result.passed,
        check=checker.result.to_json(),
        events=events.to_json(),
        fault_windows=schedule.windows(),
        cross_layer_overlaps=len(schedule.overlaps()),
        jobs=[m.to_json() for m in plane.jobs.values()],
        health_timeline=health_timeline,
        published=len(published),
        queryable=len(queryable),
        lost_uncommitted=len(managed.publisher.rejected),
        final_leader=leader,
        converged=len(set(digests.values())) <= 1,
    )
