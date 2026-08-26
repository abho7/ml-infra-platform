"""
Cross-layer fault injection.

Faults are objects consulted at each round rather than special cases inside the
control plane. That is what makes "kill a training worker while the storage
cluster is partitioned" not a scenario anybody had to write -- it is two faults
whose active windows overlap, composed by the harness.

The layer split is the whole point. Each engine's own chaos harness could only
express faults in its own layer: raft-chaos-testing can partition a cluster but
has no notion of a training worker, and the training framework can kill a worker
but has no notion of quorum. The faults below are typed by which layer they hit,
and a scenario is free to mix them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fault:
    at: int                      # round the fault fires
    until: int | None = None     # for faults with a duration

    def describe(self) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError


# ------------------------------------------------------------- storage layer


@dataclass
class KillNode(Fault):
    node: str = ""

    def describe(self) -> dict:
        return {"layer": "storage", "kind": "KillNode", "node": self.node, "at": self.at}


@dataclass
class RestartNode(Fault):
    node: str = ""

    def describe(self) -> dict:
        return {"layer": "storage", "kind": "RestartNode", "node": self.node, "at": self.at}


@dataclass
class Partition(Fault):
    groups: list[list[str]] = field(default_factory=list)

    def describe(self) -> dict:
        return {
            "layer": "storage", "kind": "Partition",
            "groups": [sorted(g) for g in self.groups],
            "at": self.at, "until": self.until,
        }


@dataclass
class DelayLink(Fault):
    """Slow a directed link without breaking it.

    Distinct from a partition and worth having separately: a slow link still
    delivers, so a leader keeps its followers but every commit takes longer.
    That is the condition under which a write times out at the client while
    still eventually committing -- the ambiguous `pending` state the publisher
    has to handle without duplicating.
    """
    src: str = ""
    dst: str = ""
    ticks: int = 5

    def describe(self) -> dict:
        return {
            "layer": "storage", "kind": "DelayLink", "src": self.src,
            "dst": self.dst, "ticks": self.ticks, "at": self.at, "until": self.until,
        }


# ------------------------------------------------------------ training layer


@dataclass
class KillWorker(Fault):
    job: str = ""
    worker: str = ""   # empty means "whichever worker is first alive"

    def describe(self) -> dict:
        return {
            "layer": "training", "kind": "KillWorker",
            "job": self.job, "worker": self.worker or "(first alive)", "at": self.at,
        }


# --------------------------------------------------------------- the schedule


class FaultSchedule:
    """A list of faults, applied to a ControlPlane round by round."""

    def __init__(self, faults: list[Fault] | None = None) -> None:
        self.faults = list(faults or [])
        self.applied: list[dict] = []
        self._link_filters: dict[int, object] = {}

    def add(self, fault: Fault) -> "FaultSchedule":
        self.faults.append(fault)
        return self

    def apply(self, plane, rnd: int) -> list[dict]:
        """Fire everything due at this round. Returns what fired."""
        fired = []
        for f in self.faults:
            if f.at != rnd:
                continue
            try:
                fired.append(self._fire(plane, f))
            except (KeyError, ValueError) as e:
                # A fault that cannot fire is recorded rather than raised: a
                # randomized schedule will sometimes name a node that is
                # already dead, and that is a property of the schedule, not an
                # error in the platform.
                fired.append({**f.describe(), "skipped": str(e)})

        # Expire duration-bounded faults.
        for f in self.faults:
            if f.until == rnd:
                if isinstance(f, Partition):
                    plane.heal()
                    fired.append({**f.describe(), "healed": True})
                elif isinstance(f, DelayLink):
                    filt = self._link_filters.pop(id(f), None)
                    if filt is not None and filt in plane.cluster.transport.filters:
                        plane.cluster.transport.filters.remove(filt)
                        fired.append({**f.describe(), "restored": True})

        self.applied.extend(fired)
        return fired

    def _fire(self, plane, f: Fault) -> dict:
        if isinstance(f, KillNode):
            plane.kill_node(f.node)
        elif isinstance(f, RestartNode):
            plane.restart_node(f.node)
        elif isinstance(f, Partition):
            plane.partition(*f.groups)
        elif isinstance(f, DelayLink):
            def filt(src, dst, _msg, _tick, _f=f):
                return _f.ticks if (src == _f.src and dst == _f.dst) else None
            plane.cluster.transport.filters.append(filt)
            self._link_filters[id(f)] = filt
        elif isinstance(f, KillWorker):
            managed = plane.job(f.job)
            worker = f.worker or next(
                (w for w in managed.job.worker_ids
                 if managed.job.engine.workers[w].alive), None
            )
            if worker is None:
                raise ValueError(f"job {f.job} has no live worker to kill")
            plane.kill_worker(f.job, worker)
            return {**f.describe(), "worker": worker}
        else:  # pragma: no cover
            raise TypeError(f"unknown fault: {type(f)!r}")
        return f.describe()

    def windows(self) -> list[dict]:
        return [f.describe() for f in self.faults]

    def overlaps(self) -> list[tuple[dict, dict]]:
        """Pairs of faults from DIFFERENT layers whose windows overlap.

        Reported on the site, because a scenario claiming to test simultaneous
        cross-layer failure should be able to show that the failures were in
        fact simultaneous and in fact in different layers.
        """
        def window(f):
            return (f.at, f.until if f.until is not None else f.at)

        out = []
        for i, a in enumerate(self.faults):
            for b in self.faults[i + 1:]:
                da, db = a.describe(), b.describe()
                if da["layer"] == db["layer"]:
                    continue
                a0, a1 = window(a)
                b0, b1 = window(b)
                if a0 <= b1 and b0 <= a1:
                    out.append((da, db))
        return out
