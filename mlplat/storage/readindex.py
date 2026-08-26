"""
ReadIndex: linearizable reads on an engine that does not provide them.

raft-engine's ARCHITECTURE.md §5 is explicit:

    "No linearizable reads. Client GETs are served locally by whichever node
     receives them, straight from its state machine, with no read-index or
     leader-lease protocol. A client could read slightly stale data from a
     lagging follower, or from an isolated former leader that doesn't yet know
     it's been superseded."

That is a real gap, and it is fixable from outside without touching the engine,
using only `RaftNode`'s public state. The protocol is the one from §8 of the
Raft paper:

    1. The node must believe it is leader. If not, refuse -- do not guess.
    2. Record its current `commit_index`. Call this the read index.
    3. Confirm leadership is still current by exchanging a heartbeat round with
       a quorum. A deposed leader cannot get a quorum, so it fails here rather
       than serving stale data.
    4. Wait until the serving replica's `last_applied >= read index`.
    5. Serve.

Step 3 is the one that matters and the one a naive implementation skips. Raft's
leader is not authoritative merely because it thinks it is: a partitioned
ex-leader still has `role == LEADER` and a perfectly plausible `commit_index`
until something tells it otherwise. Skipping the quorum round produces a read
path that looks linearizable, passes every test that does not partition the
leader, and silently serves stale data exactly when it matters.

WHAT THIS DOES NOT ADD. There is no leader lease, so every linearizable read
pays a full heartbeat round trip; a real system amortises that with a lease
and a clock bound. The cost is measured rather than hidden -- see the read
benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import mlplat.enginepath  # noqa: F401
from raft.messages import AppendEntriesResponse
from raft.node import Role


class ReadMode(str, Enum):
    LINEARIZABLE = "linearizable"
    STALE = "stale"


class ReadRefused(Exception):
    """A linearizable read could not be served safely.

    Raised rather than falling back to a stale read. A fallback would mean the
    guarantee silently degrades exactly when the cluster is in the state the
    guarantee exists for, which is worse than failing: the caller asked for
    linearizable and would receive stale data believing it was fresh.
    """


@dataclass
class ReadTicket:
    read_index: int
    issued_at_tick: int
    term: int
    confirmed: bool = False
    acks: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.acks is None:
            self.acks = set()


class ReadIndexCoordinator:
    """Issues and confirms read tickets against a StorageCluster."""

    def __init__(self, cluster) -> None:
        self.cluster = cluster
        self.issued = 0
        self.refused = 0
        self.confirm_ticks_total = 0

    # ----------------------------------------------------------------- issue

    def begin(self) -> ReadTicket:
        lid = self.cluster.leader()
        if lid is None:
            self.refused += 1
            raise ReadRefused("no leader: cannot establish a read index")

        node = self.cluster.replicas[lid].node
        if node.role != Role.LEADER:
            self.refused += 1
            raise ReadRefused(f"{lid} is not leader")

        self.issued += 1
        return ReadTicket(
            read_index=node.commit_index,
            issued_at_tick=self.cluster.tick,
            term=node.current_term,
        )

    # --------------------------------------------------------------- confirm

    def confirm(self, ticket: ReadTicket, *, max_ticks: int = 60) -> str:
        """Run the cluster until a quorum confirms this leader is still current.

        Returns the confirmed leader id. Raises ReadRefused if leadership could
        not be confirmed within `max_ticks` -- which is the correct outcome for
        a partitioned ex-leader, not a timeout bug.

        Confirmation is counted from successful AppendEntries responses at the
        ticket's term. A response is proof the responder has heard from this
        leader *in this term*, which is exactly the evidence needed.
        """
        lid = self.cluster.leader()
        if lid is None:
            self.refused += 1
            raise ReadRefused("leader disappeared before the read could be confirmed")

        leader = self.cluster.replicas[lid]
        if leader.node.current_term != ticket.term:
            self.refused += 1
            raise ReadRefused(
                f"term moved {ticket.term} -> {leader.node.current_term}; "
                "the ticket was issued by a superseded leader"
            )

        majority = len(self.cluster.node_ids) // 2 + 1
        ticket.acks = {lid}  # a leader counts toward its own quorum
        start = self.cluster.tick

        # Watch responses as they arrive rather than inspecting state after the
        # fact: match_index alone cannot distinguish "acked in this term" from
        # "acked before I was deposed and re-elected".
        observer = _AckObserver(lid, ticket)
        self.cluster.transport.filters.append(observer)
        try:
            for _ in range(max_ticks):
                if len(ticket.acks) >= majority:
                    ticket.confirmed = True
                    self.confirm_ticks_total += self.cluster.tick - start
                    return lid

                self.cluster.step()

                current = self.cluster.leader()
                if current != lid or leader.node.role != Role.LEADER:
                    self.refused += 1
                    raise ReadRefused(
                        "leadership was lost while confirming the read index"
                    )
                if leader.node.current_term != ticket.term:
                    self.refused += 1
                    raise ReadRefused("term advanced during confirmation")
        finally:
            self.cluster.transport.filters.remove(observer)

        self.refused += 1
        raise ReadRefused(
            f"could not confirm leadership within {max_ticks} ticks "
            "(no quorum reachable)"
        )

    # ------------------------------------------------------------------ wait

    def await_applied(self, node_id: str, index: int, *, max_ticks: int = 60) -> bool:
        """Run until `node_id` has applied up to `index`."""
        for _ in range(max_ticks):
            if self.cluster.replicas[node_id].sm.last_applied_index >= index:
                return True
            self.cluster.step()
        return self.cluster.replicas[node_id].sm.last_applied_index >= index

    def stats(self) -> dict:
        return {
            "issued": self.issued,
            "refused": self.refused,
            "mean_confirm_ticks": (
                self.confirm_ticks_total / self.issued if self.issued else 0.0
            ),
        }


class _AckObserver:
    """Transport filter that counts in-term AppendEntries acks to the leader.

    A filter, not a wrapper around the transport: filters already exist for
    faults and compose, so observing costs nothing structural. It never alters
    a message -- it returns None for everything.
    """

    def __init__(self, leader_id: str, ticket: ReadTicket) -> None:
        self.leader_id = leader_id
        self.ticket = ticket

    def __call__(self, sender: str, recipient: str, message, _tick):
        if (
            recipient == self.leader_id
            and isinstance(message, AppendEntriesResponse)
            and message.success
            and message.term == self.ticket.term
        ):
            self.ticket.acks.add(sender)
        return None
