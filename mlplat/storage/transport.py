"""
A queued message transport over the unmodified RaftNode.

WHY NOT raft/simulator.py. The engine ships a `SimulatedCluster`, and it is
excellent for what it was built for, but it delivers messages synchronously and
recursively: `_deliver` calls the recipient's handler immediately and recurses,
so a vote response can become a leadership change can become heartbeats, all
inside one `tick()`. A message therefore has no in-flight state, and "delay this
by five ticks" has nowhere to live. Its own docstring is explicit that it models
a zero-latency network.

This platform's whole subject is what happens when failures in different layers
overlap in time -- a storage node partitioned *while* a training worker dies,
a write in flight *when* a leader is killed. Those scenarios are unexpressible
without in-flight messages, so the transport is rebuilt here. `RaftNode` itself
is used exactly as shipped; this only changes when its outputs get delivered.

ORDERING. The queue is keyed by (deliver_at, seq) where `seq` is a monotonic
counter, not a node id. Two messages scheduled for the same tick are delivered
in the order they were sent. Ordering by id instead would make delivery order
depend on node names, which is both arbitrary and impossible for a real network
to reproduce.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable

import mlplat.enginepath  # noqa: F401
from raft.messages import (
    AppendEntriesRequest,
    AppendEntriesResponse,
    RequestVoteRequest,
    RequestVoteResponse,
)


@dataclass(order=True)
class _Queued:
    deliver_at: int
    seq: int
    sender: str = field(compare=False)
    recipient: str = field(compare=False)
    message: object = field(compare=False)


class QueuedTransport:
    """In-flight message queue with per-message fault hooks."""

    def __init__(self, *, base_latency: int = 1) -> None:
        if base_latency < 1:
            # Nothing may resolve inside the tick that produced it, or a
            # message could be sent and acted upon in the same instant -- which
            # is precisely the property that makes the engine's own simulator
            # unable to express delay.
            raise ValueError("base_latency must be at least 1 tick")
        self.base_latency = base_latency
        self._queue: list[_Queued] = []
        self._seq = itertools.count()
        self.tick = 0

        # Consulted per message, in order. Each returns None to pass, False to
        # drop, or an int to add that many ticks of delay. A list rather than a
        # single hook so several faults can be active at once and compose,
        # instead of each being a special case in the transport.
        self.filters: list[Callable] = []

        self.sent = 0
        self.delivered = 0
        self.dropped = 0
        self.delayed = 0

    # ------------------------------------------------------------------ send

    def send(self, sender: str, recipient: str, message) -> bool:
        """Queue a message. False means a fault swallowed it."""
        self.sent += 1
        delay = self.base_latency
        for f in self.filters:
            verdict = f(sender, recipient, message, self.tick)
            if verdict is None:
                continue
            if verdict is False:
                self.dropped += 1
                return False
            extra = int(verdict)
            if extra > delay:
                delay = extra
                self.delayed += 1
        heapq.heappush(
            self._queue,
            _Queued(self.tick + delay, next(self._seq), sender, recipient, message),
        )
        return True

    # --------------------------------------------------------------- deliver

    def due(self) -> list[_Queued]:
        out: list[_Queued] = []
        while self._queue and self._queue[0].deliver_at <= self.tick:
            out.append(heapq.heappop(self._queue))
        return out

    def drop_from(self, node_id: str) -> int:
        """Discard everything in flight from a node. Called when it dies.

        A crashed process cannot have packets still arriving from it. Leaving
        them queued would let a dead node keep influencing consensus, which
        would be a fault in this harness masquerading as an engine bug.
        """
        before = len(self._queue)
        self._queue = [q for q in self._queue if q.sender != node_id]
        heapq.heapify(self._queue)
        return before - len(self._queue)

    def advance(self) -> None:
        self.tick += 1

    @property
    def in_flight(self) -> int:
        return len(self._queue)

    def counters(self) -> dict:
        return {
            "sent": self.sent,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "delayed": self.delayed,
            "in_flight": self.in_flight,
        }


def dispatch(node, sender: str, message):
    """Route a message to the right RaftNode handler.

    The engine's message types are plain dataclasses with no dispatch of their
    own, so the mapping lives here rather than in the transport -- the
    transport moves bytes, this knows what they mean.
    """
    if isinstance(message, RequestVoteRequest):
        return node.handle_request_vote(sender, message)
    if isinstance(message, RequestVoteResponse):
        return node.handle_request_vote_response(sender, message)
    if isinstance(message, AppendEntriesRequest):
        return node.handle_append_entries(sender, message)
    if isinstance(message, AppendEntriesResponse):
        return node.handle_append_entries_response(sender, message)
    raise TypeError(f"unknown message type: {type(message)!r}")
