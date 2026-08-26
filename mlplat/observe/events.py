"""
One event stream for every layer.

The dashboard, the replay, the fault timelines and the chaos report are all
reconstructed from this. A single stream rather than one per layer is the whole
point: the interesting moments in this project are cross-layer -- a training
worker dying three ticks after a storage partition began -- and that
relationship is only visible if both events are on the same timeline with the
same clock.

Same discipline as the two prior projects: events are recorded as they happen,
never derived at the end, so the report can say *when* rather than *whether*.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class EventKind(str, Enum):
    # storage: consensus
    LEADER_ELECTED = "leader_elected"
    NODE_KILLED = "node_killed"
    NODE_RESTARTED = "node_restarted"
    PARTITION_STARTED = "partition_started"
    PARTITION_HEALED = "partition_healed"
    REPLICA_CAUGHT_UP = "replica_caught_up"

    # storage: client path
    WRITE_PROPOSED = "write_proposed"
    WRITE_COMMITTED = "write_committed"
    WRITE_REJECTED = "write_rejected"
    READ_SERVED = "read_served"
    READ_REFUSED = "read_refused"

    # training
    JOB_STARTED = "job_started"
    JOB_STEP = "job_step"
    JOB_FINISHED = "job_finished"
    WORKER_DIED = "worker_died"
    EMBEDDINGS_PUBLISHED = "embeddings_published"

    # control plane
    JOB_QUEUED = "job_queued"
    CLUSTER_STARTED = "cluster_started"

    # correctness
    VIOLATION = "violation"


@dataclass
class Event:
    tick: int
    kind: EventKind
    detail: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        out = asdict(self)
        out["kind"] = self.kind.value
        return out


class EventLog:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def record(self, tick: int, kind: EventKind, /, **detail) -> Event:
        """`tick` and `kind` are positional-only so callers may legitimately
        pass their own "kind" inside detail -- the violation records do."""
        ev = Event(tick=tick, kind=kind, detail=detail)
        self.events.append(ev)
        return ev

    def of_kind(self, *kinds: EventKind) -> list[Event]:
        wanted = set(kinds)
        return [e for e in self.events if e.kind in wanted]

    @property
    def violations(self) -> list[Event]:
        return self.of_kind(EventKind.VIOLATION)

    def since(self, tick: int) -> list[Event]:
        return [e for e in self.events if e.tick >= tick]

    def to_json(self) -> list[dict]:
        return [e.to_json() for e in self.events]

    def __len__(self) -> int:
        return len(self.events)
