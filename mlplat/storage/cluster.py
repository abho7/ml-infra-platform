"""
The storage cluster: N replicas, each a RaftNode plus its own vector index.

One replica = one `RaftNode` (unmodified) + one `VectorStateMachine`. Writes go
through consensus; the state machine only ever sees entries Raft has already
committed. Nothing here reaches into a node's internals to make a write happen
faster -- every write takes the real path, because a shortcut for the happy
case is exactly what would make the fault tests meaningless.

RESTART SEMANTICS follow the engine's own `SimulatedCluster.restart_node`:
term, vote and log survive; everything volatile does not. That means
`last_applied` resets to 0 and the whole committed prefix is re-delivered, so
the state machine is reset alongside it and replays from scratch. Those two
must move together -- see statemachine.py for what breaks otherwise.

The engine has no snapshot support, so catch-up is full log replay. On a long
run that is slow and the log grows forever. Stated here rather than papered
over; it is measured in the recovery benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlplat.enginepath  # noqa: F401
import numpy as np
from raft.messages import LogEntry
from raft.node import RaftNode, Role

from mlplat.observe.events import EventKind, EventLog
from mlplat.storage.statemachine import REPLICA_SEED, VectorStateMachine
from mlplat.storage.transport import QueuedTransport, dispatch


@dataclass
class Replica:
    node_id: str
    node: RaftNode
    sm: VectorStateMachine
    alive: bool = True
    # Every committed entry this replica has observed, keyed by index. Keyed
    # rather than appended because a restarted node re-reports its prefix; a
    # list would double-count it, which is the same trap the engine's own
    # simulator documents in `applied_log`.
    observed: dict[int, LogEntry] = field(default_factory=dict)


class StorageCluster:
    def __init__(
        self,
        node_ids: list[str],
        *,
        dim: int,
        seed: int = 7,
        election_timeout_ticks: tuple[int, int] = (10, 20),
        heartbeat_interval_ticks: int = 3,
        base_latency: int = 1,
        events: EventLog | None = None,
    ) -> None:
        self.node_ids = sorted(node_ids)
        self.dim = dim
        self.seed = seed
        # `is None`, not `or`. EventLog defines __len__, so a fresh empty log is
        # FALSY -- `events or EventLog()` therefore discarded the shared log the
        # caller passed in and silently created a private one. Every layer got
        # its own timeline, which is precisely the opposite of this class's
        # purpose, and nothing failed until a test inspected the shared log.
        self.events = EventLog() if events is None else events
        self.transport = QueuedTransport(base_latency=base_latency)
        self._timeouts = election_timeout_ticks
        self._heartbeat = heartbeat_interval_ticks

        self.replicas: dict[str, Replica] = {}
        for nid in self.node_ids:
            self.replicas[nid] = Replica(
                node_id=nid,
                node=RaftNode(
                    nid,
                    [p for p in self.node_ids if p != nid],
                    election_timeout_ticks=election_timeout_ticks,
                    heartbeat_interval_ticks=heartbeat_interval_ticks,
                    random_seed=f"{seed}-{nid}",
                ),
                # One shared explicit seed across every replica, so replica
                # equivalence is structural rather than empirical. See
                # statemachine.py for what that was measured to actually buy:
                # layer 0 is seed-independent, the upper layers are not, and
                # answers agree either way at this scale.
                sm=VectorStateMachine(dim=dim, seed=REPLICA_SEED),
            )

        self._partition: list[set[str]] | None = None
        self.transport.filters.append(self._partition_filter)

    # ------------------------------------------------------------ membership

    @property
    def tick(self) -> int:
        return self.transport.tick

    @property
    def live_ids(self) -> list[str]:
        return sorted(r.node_id for r in self.replicas.values() if r.alive)

    @property
    def dead_ids(self) -> list[str]:
        return sorted(r.node_id for r in self.replicas.values() if not r.alive)

    def leader(self) -> str | None:
        """The single current leader, or None.

        Raft guarantees at most one leader *per term*, not at most one leader
        at any instant: a deposed leader is entitled not to know yet. When more
        than one node claims leadership, the one with the highest term is the
        real one.
        """
        claims = [
            r for r in self.replicas.values()
            if r.alive and r.node.role == Role.LEADER
        ]
        if not claims:
            return None
        return max(claims, key=lambda r: r.node.current_term).node_id

    def leader_claims(self) -> list[str]:
        return sorted(
            r.node_id for r in self.replicas.values()
            if r.alive and r.node.role == Role.LEADER
        )

    # ---------------------------------------------------------------- faults

    def kill(self, node_id: str) -> int:
        r = self.replicas[node_id]
        if not r.alive:
            return 0
        r.alive = False
        lost = self.transport.drop_from(node_id)
        self.events.record(
            self.tick, EventKind.NODE_KILLED, node=node_id, inflight_lost=lost
        )
        return lost

    def restart(self, node_id: str) -> None:
        """Crash-restart with the engine's own durability contract.

        Term, vote and log survive; role, commit_index, last_applied and the
        index do not. The state machine is reset in the same breath so the
        replica replays its committed prefix from scratch rather than believing
        it is caught up while holding an empty graph.
        """
        r = self.replicas[node_id]
        old = r.node
        fresh = RaftNode(
            node_id,
            old.peer_ids,
            election_timeout_ticks=self._timeouts,
            heartbeat_interval_ticks=self._heartbeat,
            random_seed=f"restart-{self.tick}-{node_id}",
        )
        fresh.current_term = old.current_term
        fresh.voted_for = old.voted_for
        fresh.log = old.log

        r.node = fresh
        r.sm.reset()
        r.observed.clear()
        r.alive = True
        self.events.record(self.tick, EventKind.NODE_RESTARTED, node=node_id)

    def partition(self, *groups: set[str]) -> None:
        covered: set[str] = set()
        for g in groups:
            if covered & g:
                raise ValueError("partition groups must be disjoint")
            covered |= g
        if covered != set(self.node_ids):
            raise ValueError("partition groups must cover every node")
        self._partition = [set(g) for g in groups]
        self.events.record(
            self.tick, EventKind.PARTITION_STARTED,
            groups=[sorted(g) for g in self._partition],
        )

    def heal(self) -> None:
        if self._partition is not None:
            self.events.record(self.tick, EventKind.PARTITION_HEALED)
        self._partition = None

    def _partition_filter(self, sender: str, recipient: str, _msg, _tick):
        if not self.replicas[sender].alive or not self.replicas[recipient].alive:
            return False
        if self._partition is None:
            return None
        for g in self._partition:
            if sender in g:
                return None if recipient in g else False
        return None

    # --------------------------------------------------------------- driving

    def step(self) -> None:
        """One tick: deliver what is due, then tick every live node."""
        for q in self.transport.due():
            r = self.replicas.get(q.recipient)
            if r is None or not r.alive:
                continue
            self.transport.delivered += 1
            self._absorb(q.recipient, dispatch(r.node, q.sender, q.message))

        for nid in self.node_ids:
            r = self.replicas[nid]
            if r.alive:
                self._absorb(nid, r.node.tick())

        self.transport.advance()

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.step()

    def _absorb(self, node_id: str, actions) -> None:
        """Apply a node's Actions: commit entries locally, queue its messages."""
        r = self.replicas[node_id]
        for entry in actions.committed_entries:
            r.observed[entry.index] = entry
            r.sm.apply(entry)
        if actions.became_leader:
            self.events.record(
                self.tick, EventKind.LEADER_ELECTED,
                node=node_id, term=r.node.current_term,
            )
        for recipient, message in actions.messages:
            self.transport.send(node_id, recipient, message)

    def run_until_leader(self, max_ticks: int = 300) -> str | None:
        for _ in range(max_ticks):
            self.step()
            if self.leader() is not None and len(self.leader_claims()) == 1:
                return self.leader()
        return self.leader()

    # --------------------------------------------------------------- writing

    def propose(self, command: dict) -> tuple[str, int] | None:
        """Propose through the current leader. Returns (leader_id, log_index).

        None means there is no leader to accept it right now -- during an
        election, or on the minority side of a partition. The caller decides
        whether to retry; silently swallowing that would turn "the cluster
        could not accept this write" into "the write vanished".
        """
        lid = self.leader()
        if lid is None:
            return None
        r = self.replicas[lid]
        entry, actions = r.node.propose(command)
        if entry is None:
            return None
        self._absorb(lid, actions)
        return lid, entry.index

    def committed_on(self, node_id: str) -> int:
        return self.replicas[node_id].node.commit_index

    def quorum_committed(self, index: int) -> bool:
        """True once a majority of nodes hold `index` in their log AND the
        leader's commit_index covers it. Both halves matter: replication
        without commitment is not durability."""
        majority = len(self.node_ids) // 2 + 1
        have = sum(
            1 for r in self.replicas.values()
            if r.node.log.last_index >= index
        )
        lid = self.leader()
        committed = lid is not None and self.replicas[lid].node.commit_index >= index
        return have >= majority and committed

    def run_until_committed(self, index: int, max_ticks: int = 300) -> bool:
        for _ in range(max_ticks):
            if self.quorum_committed(index):
                return True
            self.step()
        return self.quorum_committed(index)

    # --------------------------------------------------------------- reading

    def search_on(self, node_id: str, query: np.ndarray, k: int) -> list[dict]:
        return self.replicas[node_id].sm.search(query, k)

    def digests(self) -> dict[str, tuple]:
        return {
            nid: r.sm.digest()
            for nid, r in sorted(self.replicas.items())
            if r.alive
        }

    def state(self) -> dict:
        """A snapshot for the control plane and dashboard."""
        return {
            "tick": self.tick,
            "leader": self.leader(),
            "nodes": {
                nid: {
                    "alive": r.alive,
                    "role": r.node.role.value,
                    "term": r.node.current_term,
                    "log_len": r.node.log.last_index,
                    "commit_index": r.node.commit_index,
                    "applied_index": r.sm.last_applied_index,
                    "vectors": r.sm.size,
                }
                for nid, r in sorted(self.replicas.items())
            },
            "partition": [sorted(g) for g in self._partition] if self._partition else None,
            "transport": self.transport.counters(),
        }
