"""
Writes a training job's embeddings into the storage cluster.

The seam between the two halves of the platform, and the place where the
project's central question gets answered concretely: what happens to a model's
output when the thing storing it fails mid-write.

ONE RULE, and everything else follows from it: this uses `StorageClient.write`
and nothing else. It never touches a replica's state machine directly and never
proposes to a RaftNode itself. Every embedding takes the full consensus path,
because a fast path for the happy case is exactly what would make the fault
scenarios prove nothing.

WHAT COUNTS AS PUBLISHED. A write returns `committed=True` only once the entry
is quorum-committed under the current leader's term. Anything else is recorded
as *not* published, with the reason kept. The distinction matters more than it
looks:

  rejected     no leader would accept it. The entry does not exist. Safe to
               retry, and retried.
  uncommitted  proposed, but commitment was not observed within the timeout.
               This is the genuinely ambiguous state -- the entry may yet
               commit under a future leader. It is NOT retried, because
               retrying an entry that later commits is how you get the same
               embedding stored twice.

That second case is the reason `pending` exists separately from `failed`. A
publisher that collapsed them into "failed, retry" would silently duplicate
data exactly when the cluster was already in trouble.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mlplat.observe.events import EventKind
from mlplat.storage.client import StorageClient


@dataclass
class PublishReport:
    checkpoint_step: int
    attempted: int = 0
    committed: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    pending: list[tuple[str, int]] = field(default_factory=list)
    ticks: int = 0

    @property
    def all_committed(self) -> bool:
        return len(self.committed) == self.attempted

    def to_json(self) -> dict:
        return {
            "checkpoint_step": self.checkpoint_step,
            "attempted": self.attempted,
            "committed": len(self.committed),
            "rejected": [{"id": i, "reason": r} for i, r in self.rejected],
            "pending": [{"id": i, "index": x} for i, x in self.pending],
            "ticks": self.ticks,
        }


class EmbeddingPublisher:
    def __init__(self, client: StorageClient, *, events=None) -> None:
        self.client = client
        # See the note in cluster.py: an empty EventLog is falsy.
        self.events = client.cluster.events if events is None else events
        # Every embedding this publisher was ever told was durable. The
        # cross-layer checker compares this against what the cluster can
        # actually serve; the two disagreeing is the headline failure this
        # whole platform exists to detect.
        self.published: dict[str, int] = {}
        self.pending: dict[str, int] = {}
        self.rejected: dict[str, str] = {}

    def publish(self, checkpoint) -> PublishReport:
        start = self.client.cluster.tick
        rep = PublishReport(checkpoint_step=checkpoint.step,
                            attempted=len(checkpoint.ids))

        for eid, vec, meta in zip(checkpoint.ids, checkpoint.vectors, checkpoint.metadata):
            res = self.client.write(eid, vec, meta)
            if res.committed:
                self.published[eid] = res.index
                rep.committed.append(eid)
            elif res.index is not None:
                # Proposed but unconfirmed. Deliberately not retried.
                self.pending[eid] = res.index
                rep.pending.append((eid, res.index))
            else:
                self.rejected[eid] = res.reason or "unknown"
                rep.rejected.append((eid, res.reason or "unknown"))

        rep.ticks = self.client.cluster.tick - start
        self.events.record(
            self.client.cluster.tick, EventKind.EMBEDDINGS_PUBLISHED,
            step=checkpoint.step, attempted=rep.attempted,
            committed=len(rep.committed), pending=len(rep.pending),
            rejected=len(rep.rejected), ticks=rep.ticks,
        )
        return rep

    # ------------------------------------------------------------ resolution

    def resolve_pending(self) -> dict[str, str]:
        """Decide what became of the ambiguous writes, once things settle.

        An entry proposed but not confirmed either committed later under a new
        leader or was truncated away when a different leader overwrote its log
        position. Both are correct Raft outcomes. Which one happened is only
        knowable after the cluster stabilises, and until then the honest answer
        is "unknown" -- so this is called at the end of a run rather than
        guessed at write time.
        """
        out: dict[str, str] = {}
        for eid, index in list(self.pending.items()):
            leader = self.client.cluster.leader()
            if leader is None:
                out[eid] = "unknown: no leader"
                continue
            if self.client.cluster.replicas[leader].sm.contains(eid):
                # It landed after all. Promote it, so the accounting reflects
                # reality rather than what was known at write time.
                self.published[eid] = index
                self.pending.pop(eid)
                out[eid] = "committed late"
            else:
                self.pending.pop(eid)
                self.rejected[eid] = "truncated by a later leader"
                out[eid] = "lost (uncommitted, correctly)"
        return out

    def stats(self) -> dict:
        return {
            "published": len(self.published),
            "pending": len(self.pending),
            "rejected": len(self.rejected),
        }
