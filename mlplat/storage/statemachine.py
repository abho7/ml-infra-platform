"""
The Raft state machine that owns an HNSW index.

This is the seam where the two engines meet, and it is where the first real
composition hazard lives. Both engines are individually correct and their
requirements are directly incompatible at the naive wiring:

  RAFT REQUIRES apply() to be idempotent. `last_applied` is volatile state
  that resets to 0 on restart, so a rejoining node legitimately re-delivers
  its entire pre-crash prefix as "newly committed". raft-engine's own
  kvstore/state_machine.py says so explicitly and handles it with an index
  watermark.

  HNSW REFUSES duplicates. `VectorDB.insert` raises ValueError if the id
  already exists, and `HNSWIndex.insert` raises on a duplicate node_id.

Wire them together without a watermark and every replica that restarts crashes
on the first re-delivered entry. The watermark below is not defensive
programming; it is the required adapter between two correct components.

THE SECOND HAZARD is subtler and has no analogue in the KV store. The index and
its watermark are two pieces of state that must be reset as ONE unit:

  * watermark kept, index dropped -> a replica that believes it is fully
    caught up while holding nothing. Silent, total data loss, and the node
    reports itself healthy.
  * index kept, watermark reset   -> duplicate-insert crash on the first
    re-delivered entry.

`reset()` exists so there is exactly one way to do it, and invariant 3 in the
chaos checker verifies the two never disagree. A KV state machine survives
getting this wrong because assigning a dict key twice is harmless; an HNSW
graph does not.
"""

from __future__ import annotations

import numpy as np

import mlplat.enginepath  # noqa: F401  (sys.path side effect)
from raft.messages import LogEntry
from vectordb.store import VectorDB

# Every replica in a cluster is built with this one explicit seed, because
# `random.Random(None)` seeds from system entropy and replica equivalence
# should be structural rather than accidental.
#
# What that actually buys, measured rather than assumed (see
# tests/test_storage.py::test_seed_changes_routing_but_not_answers). At the
# scales this platform runs:
#
#   * LAYER 0 IS IDENTICAL regardless of seed. It contains every node and its
#     edges come from the distance-based diversity heuristic, with no RNG
#     involved, so it is fully determined by insertion order -- which Raft
#     already guarantees is identical across replicas.
#   * UPPER LAYERS DIFFER. The per-insert level draw is the only use of the
#     RNG. At n=300, seed 1 produced layers [300, 26, 1] and seed 999 produced
#     [300, 16], with different entry points.
#   * SEARCH RESULTS AGREE ANYWAY. Across 480 query comparisons at n=60/300/600
#     and ef_search from 1 to 16, differently-seeded indexes returned identical
#     results every time: the upper layers only choose where the layer-0 search
#     starts, and a dense identical layer 0 converges to the same answer.
#
# So the shared seed is a DEFENSIVE requirement, not a demonstrated hazard. It
# is kept because "any replica answers identically" should rest on the replicas
# being identical by construction rather than on an empirical property of small
# indexes that may not survive a larger n or a higher dimension.
REPLICA_SEED = 20260825


class VectorStateMachine:
    """Deterministic replicated state machine over a vector index.

    State is fully determined by replaying the committed log in order, which is
    what makes replica convergence checkable: same log, same seed, same graph.
    """

    def __init__(
        self,
        *,
        dim: int,
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
        seed: int = REPLICA_SEED,
    ) -> None:
        self._dim = dim
        self._metric = metric
        self._M = M
        self._ef_construction = ef_construction
        self._seed = seed

        self._db: VectorDB
        self._last_applied_index: int
        self._ids_in_order: list[str]
        self._ids_seen: set[str]
        self._duplicates_suppressed: int
        self.reset()

    # ------------------------------------------------------------------ reset

    def reset(self) -> None:
        """Drop the index and the watermark together.

        The only supported way to clear this state machine. Called on replica
        restart, where Raft's own contract is that volatile state goes and the
        log is replayed from the beginning. Resetting either half alone is the
        hazard described in the module docstring.
        """
        self._db = VectorDB(
            dim=self._dim,
            metric=self._metric,
            M=self._M,
            ef_construction=self._ef_construction,
            seed=self._seed,
        )
        self._last_applied_index = 0
        self._ids_in_order = []
        self._ids_seen = set()
        self._duplicates_suppressed = 0

    # ------------------------------------------------------------------ apply

    def apply(self, entry: LogEntry) -> bool:
        """Apply one committed entry. Returns True if it changed state.

        False means the entry was at or below the watermark -- already applied,
        idempotent no-op. That is the normal, correct path for a rejoining
        node, not an error.
        """
        if entry.index <= self._last_applied_index:
            return False

        op = entry.command.get("op")
        if op == "insert":
            eid = entry.command["id"]
            if eid in self._ids_seen:
                # The SAME id at a DIFFERENT log index. The watermark cannot
                # catch this -- both indices are above it, so both look new.
                #
                # Found by the cross-layer chaos harness: a write that was
                # proposed, timed out unconfirmed, and got re-proposed can
                # commit twice. The client no longer does that (see
                # client.py), but this guard is the load-bearing half of the
                # fix and must stay regardless.
                #
                # WHY THIS MUST NOT RAISE. Every replica applies the same
                # committed log. An exception here does not reject one bad
                # write -- it stops that replica applying anything further,
                # while its peers carry on. The replica then silently diverges
                # and still reports itself alive. A state machine over a
                # replicated log has to be TOTAL: every committed entry must
                # produce a defined state transition, even if that transition
                # is "nothing". Duplicate suppression is that transition.
                self._duplicates_suppressed += 1
                self._last_applied_index = entry.index
                return False
            vec = np.asarray(entry.command["vector"], dtype=np.float64)
            self._db.insert(eid, vec, entry.command.get("metadata") or {})
            self._ids_in_order.append(eid)
            self._ids_seen.add(eid)
        elif op == "delete":
            # Tolerant of a delete for an id that was never inserted: the log
            # is the source of truth for ordering, and refusing here would make
            # a replica diverge from its peers over a no-op.
            try:
                self._db.delete(entry.command["id"])
            except KeyError:
                pass
        elif op == "noop":
            pass
        else:
            raise ValueError(f"unknown command op: {op!r}")

        # Advanced only AFTER the operation succeeded. Advancing first would
        # leave the watermark claiming an entry that raised, which is exactly
        # the index/watermark disagreement invariant 3 exists to catch.
        self._last_applied_index = entry.index
        return True

    def apply_all(self, entries: list[LogEntry]) -> int:
        return sum(1 for e in entries if self.apply(e))

    # ------------------------------------------------------------------ reads

    def search(self, query: np.ndarray, k: int, ef_search: int | None = None) -> list[dict]:
        return self._db.search(np.asarray(query, dtype=np.float64), k, ef_search)

    def get_metadata(self, external_id: str) -> dict:
        return self._db.get_metadata(external_id)

    def contains(self, external_id: str) -> bool:
        try:
            self._db.get_metadata(external_id)
            return True
        except KeyError:
            return False

    # ------------------------------------------------------------------ state

    @property
    def last_applied_index(self) -> int:
        return self._last_applied_index

    @property
    def duplicates_suppressed(self) -> int:
        """Committed entries whose id was already present.

        Surfaced rather than swallowed: a non-zero count means something
        upstream proposed the same embedding twice, which is worth knowing even
        though the state machine handled it correctly.
        """
        return self._duplicates_suppressed

    @property
    def size(self) -> int:
        return len(self._db)

    @property
    def ids_in_order(self) -> list[str]:
        """Ids in the order they were applied. Two replicas that applied the
        same committed prefix must agree on this exactly -- it is the cheapest
        structural check for apply-order drift, and unlike a search comparison
        it does not depend on the graph being queried."""
        return list(self._ids_in_order)

    def digest(self) -> tuple[int, int, tuple[str, ...]]:
        """A comparable fingerprint of this replica's applied state."""
        return (self._last_applied_index, self.size, tuple(self._ids_in_order))
