"""
The storage layer's public API.

Everything above this line (training, control plane, dashboard) talks to the
cluster through here. Two operations, and the honesty of the whole system rests
on how precisely their guarantees are stated:

  write(id, vector, metadata) -> WriteResult
      Linearizable. Returns only once the entry is quorum-committed under the
      current leader's term. A write that could not be accepted returns
      committed=False with a reason; it is never silently dropped and never
      reported as succeeded.

  search(query, k, mode)      -> SearchResult
      mode=LINEARIZABLE  reflects every write acked before the call began.
                         Costs a heartbeat quorum round. Raises ReadRefused on
                         the minority side of a partition.
      mode=STALE         served from a replica's local index with no
                         coordination. May lag arbitrarily. Explicit opt-in.

The default is LINEARIZABLE. A system whose safe mode is opt-in gets used
unsafely, and the entire reason for implementing ReadIndex was to be able to
offer the strong guarantee rather than inherit the weak one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mlplat.observe.events import EventKind
from mlplat.storage.readindex import ReadIndexCoordinator, ReadMode, ReadRefused


@dataclass
class WriteResult:
    id: str
    committed: bool
    index: int | None = None
    leader: str | None = None
    ticks: int = 0
    reason: str | None = None


@dataclass
class SearchResult:
    hits: list[dict] = field(default_factory=list)
    mode: str = ReadMode.LINEARIZABLE.value
    served_by: str | None = None
    read_index: int | None = None
    ticks: int = 0


class StorageClient:
    def __init__(self, cluster, *, commit_timeout: int = 200) -> None:
        self.cluster = cluster
        self.reader = ReadIndexCoordinator(cluster)
        self.commit_timeout = commit_timeout
        # Ids this client has been told are durable. The training layer and the
        # chaos checker both need this: "was this write acknowledged" is a
        # different question from "is it in some replica's log", and only the
        # first one carries a promise.
        self.acked: dict[str, int] = {}

    # ----------------------------------------------------------------- write

    def write(
        self,
        external_id: str,
        vector: np.ndarray,
        metadata: dict | None = None,
        *,
        retries: int = 3,
    ) -> WriteResult:
        """Propose an insert and wait for quorum commit.

        Retries exist because "there is no leader right now" is a normal,
        transient state during an election, not a failure. It is distinguished
        from "this cluster cannot accept writes" by bounding the attempts and
        returning the reason rather than blocking forever.
        """
        command = {
            "op": "insert",
            "id": external_id,
            "vector": np.asarray(vector, dtype=np.float64).tolist(),
            "metadata": metadata or {},
        }
        start = self.cluster.tick

        for attempt in range(retries):
            proposed = self.cluster.propose(command)
            if proposed is None:
                self.cluster.events.record(
                    self.cluster.tick, EventKind.WRITE_REJECTED,
                    id=external_id, reason="no leader", attempt=attempt,
                )
                # Give an election a chance to resolve before giving up.
                self.cluster.run_until_leader(max_ticks=60)
                continue

            leader_id, index = proposed
            self.cluster.events.record(
                self.cluster.tick, EventKind.WRITE_PROPOSED,
                id=external_id, index=index, leader=leader_id,
            )

            if self.cluster.run_until_committed(index, max_ticks=self.commit_timeout):
                ticks = self.cluster.tick - start
                self.acked[external_id] = index
                self.cluster.events.record(
                    self.cluster.tick, EventKind.WRITE_COMMITTED,
                    id=external_id, index=index, leader=leader_id, ticks=ticks,
                )
                return WriteResult(
                    id=external_id, committed=True, index=index,
                    leader=leader_id, ticks=ticks,
                )

            # Proposed but not committed within the timeout. RETURN -- do not
            # loop round and propose it again.
            #
            # This loop used to fall through here, and it was a real bug found
            # by the cross-layer chaos harness (partition + worker kill, seed
            # 7). The entry IS in the leader's log; it simply has not been
            # confirmed. Re-proposing appends a SECOND entry carrying the same
            # embedding id, and if the partition heals such that both commit,
            # the same id arrives at two different log indices. The state
            # machine's watermark dedupes by index, so both look new, and the
            # second insert raises out of VectorDB -- killing apply on every
            # replica at once.
            #
            # Neither engine is at fault: Raft committed two entries because it
            # was asked to, and HNSW refused a duplicate because it should. The
            # defect is here, in treating "unconfirmed" as "safe to retry".
            # Only a proposal that appended NOTHING (propose returned None) can
            # be retried safely.
            self.cluster.events.record(
                self.cluster.tick, EventKind.WRITE_REJECTED,
                id=external_id, index=index, reason="not committed within timeout",
            )
            return WriteResult(
                id=external_id, committed=False, index=index, leader=leader_id,
                ticks=self.cluster.tick - start,
                reason="proposed but not committed within the timeout",
            )

        return WriteResult(
            id=external_id, committed=False, ticks=self.cluster.tick - start,
            reason="no leader accepted the write",
        )

    # ---------------------------------------------------------------- search

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        *,
        mode: ReadMode = ReadMode.LINEARIZABLE,
        node_id: str | None = None,
    ) -> SearchResult:
        start = self.cluster.tick

        if mode is ReadMode.STALE:
            target = node_id or (self.cluster.live_ids[0] if self.cluster.live_ids else None)
            if target is None:
                raise ReadRefused("no live replica to serve a stale read")
            hits = self.cluster.search_on(target, query, k)
            self.cluster.events.record(
                self.cluster.tick, EventKind.READ_SERVED,
                mode="stale", node=target, hits=len(hits),
            )
            return SearchResult(
                hits=hits, mode=mode.value, served_by=target,
                ticks=self.cluster.tick - start,
            )

        try:
            ticket = self.reader.begin()
            leader_id = self.reader.confirm(ticket)
        except ReadRefused as e:
            self.cluster.events.record(
                self.cluster.tick, EventKind.READ_REFUSED,
                mode="linearizable", reason=str(e),
            )
            raise

        target = node_id or leader_id
        if not self.reader.await_applied(target, ticket.read_index):
            self.cluster.events.record(
                self.cluster.tick, EventKind.READ_REFUSED,
                mode="linearizable", reason="replica did not catch up to the read index",
            )
            raise ReadRefused(
                f"{target} did not reach applied index {ticket.read_index}"
            )

        hits = self.cluster.search_on(target, query, k)
        self.cluster.events.record(
            self.cluster.tick, EventKind.READ_SERVED,
            mode="linearizable", node=target,
            read_index=ticket.read_index, hits=len(hits),
        )
        return SearchResult(
            hits=hits, mode=mode.value, served_by=target,
            read_index=ticket.read_index, ticks=self.cluster.tick - start,
        )

    # ------------------------------------------------------------------ misc

    def stats(self) -> dict:
        return {"acked_writes": len(self.acked), "reads": self.reader.stats()}
