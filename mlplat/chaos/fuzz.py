"""
Randomized cross-layer fault sweep.

The eight named scenarios are hand-built, which makes them good report material
and poor bug hunters: they only probe the interleavings somebody thought of.
This generates random schedules mixing storage and training faults, so "no
violation found" is backed by search rather than by eight lucky orderings.

Every source of randomness is seeded, so any violating seed prints and replays
exactly. That guarantee is the reason a failure here is actionable rather than
a rumour, and it is the same one raft-chaos-testing's sweep makes.

Run:  python -m mlplat.chaos.fuzz [n_seeds]
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass

from mlplat.chaos.faults import (
    DelayLink, FaultSchedule, KillNode, KillWorker, Partition, RestartNode,
)
from mlplat.chaos.runner import run


@dataclass
class FuzzOutcome:
    seed: int
    violations: list[dict]
    actions: list[str]
    rounds: int
    published: int
    queryable: int
    converged: bool
    accuracy: float | None

    @property
    def passed(self) -> bool:
        return not self.violations


def build_schedule(seed: int, node_ids: list[str], steps: int
                   ) -> tuple[FaultSchedule, list[str]]:
    """A random but reproducible cross-layer schedule.

    Bounded so a majority always remains reachable at the end. A run where the
    cluster can never commit again is an availability outcome, not a
    correctness one -- letting those in would turn a meaningful failure count
    into noise about scenarios that were unwinnable by construction.
    """
    rng = random.Random(seed * 7919 + 13)
    faults, actions = [], []
    n = len(node_ids)
    max_kills = (n - 1) // 2  # never break quorum permanently

    doomed = rng.sample(node_ids, rng.randint(0, max_kills))
    for nid in doomed:
        at = rng.randint(4, max(5, steps // 2))
        faults.append(KillNode(at=at, node=nid))
        actions.append(f"kill {nid}@{at}")
        if rng.random() < 0.5:
            back = at + rng.randint(10, 30)
            faults.append(RestartNode(at=back, node=nid))
            actions.append(f"restart {nid}@{back}")

    if rng.random() < 0.6:
        start = rng.randint(5, max(6, steps // 2))
        end = start + rng.randint(8, 30)
        survivors = [x for x in node_ids if x not in doomed]
        if len(survivors) >= 2:
            k = rng.randint(1, len(survivors) - 1)
            a = set(rng.sample(survivors, k))
            groups = [sorted(a), sorted(set(node_ids) - a)]
            faults.append(Partition(at=start, until=end, groups=groups))
            actions.append(f"partition {groups[0]}|{groups[1]} {start}-{end}")

    for _ in range(rng.randint(0, 2)):
        at = rng.randint(4, max(5, steps - 5))
        faults.append(KillWorker(at=at, job="job"))
        actions.append(f"kill-worker@{at}")

    if rng.random() < 0.4 and n >= 2:
        a, b = rng.sample(node_ids, 2)
        start = rng.randint(4, max(5, steps // 2))
        faults.append(DelayLink(at=start, until=start + rng.randint(6, 25),
                                src=a, dst=b, ticks=rng.randint(3, 10)))
        actions.append(f"delay {a}->{b}@{start}")

    return FaultSchedule(faults), actions


def run_one(seed: int, *, steps: int = 40) -> FuzzOutcome:
    rng = random.Random(seed * 104729 + 7)
    n_nodes = rng.choice([3, 5, 5, 7])
    n_workers = rng.randint(2, 5)
    node_ids = [f"n{i + 1}" for i in range(n_nodes)]
    schedule, actions = build_schedule(seed, node_ids, steps)

    r = run(
        name=f"fuzz-{seed}", slug=f"fuzz-{seed}",
        description="randomized cross-layer schedule",
        fault_summary="; ".join(actions) or "none",
        schedule=schedule, n_nodes=n_nodes, n_workers=n_workers,
        steps=steps, seed=seed, publish_every=10,
    )
    job = r.jobs[0]
    return FuzzOutcome(
        seed=seed, violations=r.check["violations"], actions=actions,
        rounds=r.rounds, published=r.published, queryable=r.queryable,
        converged=r.converged, accuracy=job["accuracy"],
    )


def sweep(n_seeds: int = 100, *, quiet: bool = False) -> dict:
    outcomes = []
    for seed in range(1, n_seeds + 1):
        o = run_one(seed)
        outcomes.append(o)
        if not o.passed and not quiet:
            print(f"  SEED {o.seed} VIOLATED: {o.actions}")
            for v in o.violations[:4]:
                print(f"    {v}")

    failed = [o for o in outcomes if not o.passed]
    unconverged = [o for o in outcomes if not o.converged]
    unqueryable = [o for o in outcomes if o.published != o.queryable]

    return {
        "seeds": n_seeds,
        "runs": len(outcomes),
        "passed": len(outcomes) - len(failed),
        "failed": len(failed),
        "failing_seeds": [
            {"seed": o.seed, "actions": o.actions, "violations": o.violations}
            for o in failed
        ],
        "unconverged": len(unconverged),
        # Published-but-not-queryable is the headline cross-layer failure: the
        # platform told a training job its embedding was durable and then could
        # not serve it back.
        "published_not_queryable": len(unqueryable),
        "total_published": sum(o.published for o in outcomes),
        "min_accuracy": min((o.accuracy for o in outcomes if o.accuracy is not None),
                            default=0.0),
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f"Fuzzing {n} cross-layer seeds...")
    s = sweep(n)
    print(
        f"\n{s['runs']} runs: {s['passed']} passed, {s['failed']} failed\n"
        f"embeddings published: {s['total_published']}\n"
        f"published but not queryable: {s['published_not_queryable']}\n"
        f"clusters that failed to converge: {s['unconverged']}\n"
        f"lowest final accuracy: {s['min_accuracy']:.4f}"
    )
    sys.exit(1 if s["failed"] else 0)
