"""
The cross-layer correctness checker.

Every invariant here lives at a SEAM. Raft's own safety properties are proven
by its 23 tests, HNSW's recall by its 28, gradient exactness by the training
engine's 68. None of those can be violated by anything this platform does,
because none of those engines is modified. What can be violated is the
composition -- and that is a genuinely different claim, because the failure
modes only exist once two correct components are wired together.

The six properties, each checked continuously rather than at the end so the
report can say *when*:

  1 ACKED-WRITE DURABILITY   an embedding the publisher was told was durable is
                             present on every subsequently elected leader
  2 REPLICA CONVERGENCE      replicas agree on their common applied prefix
                             always, and on everything once quiesced
  3 INDEX/WATERMARK AGREEMENT a replica's index holds exactly the committed
                             inserts it claims to have applied
  4 NO PHANTOM EMBEDDINGS    nothing is searchable that was not committed
                             through Raft
  5 EXACTLY-ONCE ACCOUNTING  no embedding is stored twice
  6 READ HONESTY             a linearizable read never returns a state older
                             than a write acked before it began

DELIBERATELY NOT VIOLATIONS, recorded and reported separately:

  * a stale read lagging -- that is the mode's contract, not a defect
  * a minority-side linearizable read failing -- correct, and the entire point
    of implementing ReadIndex
  * an un-acknowledged in-flight write vanishing when a node dies -- correct,
    and the direct analogue of an uncommitted Raft entry
  * a training job's accuracy dropping after worker loss -- a smaller effective
    batch is the consequence of the fault, not a bug

Counting any of those as failures would bury the real signal, which is the same
argument raft-chaos-testing makes for refusing to count stale follower reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mlplat.observe.events import EventKind, EventLog


class ViolationKind:
    DURABILITY = "acked_write_durability"
    CONVERGENCE = "replica_convergence"
    INDEX_WATERMARK = "index_watermark_agreement"
    PHANTOM = "phantom_embedding"
    EXACTLY_ONCE = "exactly_once_accounting"
    READ_HONESTY = "linearizable_read_honesty"


@dataclass
class CheckResult:
    passed: bool = True
    violations: list[dict] = field(default_factory=list)
    checks_run: int = 0

    # Reported, never failed on.
    stale_reads_lagging: int = 0
    minority_reads_refused: int = 0
    uncommitted_writes_lost: int = 0
    max_replica_lag: int = 0
    # Observation, not a pass/fail: how much of the acked set an approximate
    # k-NN query happened to return. See check_read_honesty for why this is
    # recorded rather than asserted.
    read_recall_samples: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "checks_run": self.checks_run,
            "stale_reads_lagging": self.stale_reads_lagging,
            "minority_reads_refused": self.minority_reads_refused,
            "uncommitted_writes_lost": self.uncommitted_writes_lost,
            "max_replica_lag": self.max_replica_lag,
            "read_recall_samples": self.read_recall_samples,
        }


class CrossLayerChecker:
    """Attach to a ControlPlane; call `check()` every round."""

    def __init__(self, plane, *, events: EventLog | None = None) -> None:
        self.plane = plane
        self.cluster = plane.cluster
        self.events = plane.events if events is None else events
        self.result = CheckResult()
        # Every id ever seen applied on any replica, with the log index it came
        # from. Invariant 5 is a statement about this map, not about a single
        # replica: an id applied at two DIFFERENT indices is a duplicate even
        # if no single replica holds it twice.
        self._id_index: dict[str, int] = {}
        self._leaders_seen: set[tuple[str, int]] = set()

    # ------------------------------------------------------------- reporting

    def _violate(self, kind: str, detail: str, **extra) -> None:
        self.result.passed = False
        rec = {"tick": self.cluster.tick, "kind": kind, "detail": detail, **extra}
        self.result.violations.append(rec)
        self.events.record(
            self.cluster.tick, EventKind.VIOLATION, kind=kind, detail=detail, **extra
        )

    # ---------------------------------------------------------------- checks

    def check(self) -> None:
        self.result.checks_run += 1
        self._check_durability()
        self._check_prefix_agreement()
        self._check_index_watermark()
        self._check_phantoms_and_duplicates()

    # 1 ------------------------------------------------------------------

    def _check_durability(self):
        """Every acked embedding must survive every leadership change.

        Checked when the leader changes rather than every tick: that is the
        only moment durability can actually be lost, and checking on a stable
        leader would just re-assert the same fact thousands of times.
        """
        leader = self.cluster.leader()
        if leader is None:
            return
        key = (leader, self.cluster.replicas[leader].node.current_term)
        if key in self._leaders_seen:
            return
        self._leaders_seen.add(key)

        for managed in self.plane.jobs.values():
            for eid in managed.publisher.published:
                if not self.cluster.replicas[leader].sm.contains(eid):
                    # A newly elected leader may not have APPLIED the entry
                    # yet even though its log holds it. Only a missing log
                    # entry is a durability failure.
                    log = self.cluster.replicas[leader].node.log
                    idx = managed.publisher.published[eid]
                    if log.last_index < idx:
                        self._violate(
                            ViolationKind.DURABILITY,
                            f"acked embedding {eid} (index {idx}) is absent from "
                            f"new leader {leader}'s log",
                            id=eid, index=idx, leader=leader,
                        )

    # 2 ------------------------------------------------------------------

    def _check_prefix_agreement(self):
        """Replicas must agree on their common applied prefix at every instant.

        Full convergence is only guaranteed after quiescence -- followers
        legitimately lag while commit_index propagates -- so the always-true
        invariant is prefix agreement. `check_converged` asserts the stronger
        one where it applies.
        """
        orders = {
            nid: r.sm.ids_in_order
            for nid, r in self.cluster.replicas.items() if r.alive
        }
        if len(orders) < 2:
            return
        lengths = [len(v) for v in orders.values()]
        self.result.max_replica_lag = max(
            self.result.max_replica_lag, max(lengths) - min(lengths)
        )
        m = min(lengths)
        prefixes = {nid: tuple(v[:m]) for nid, v in orders.items()}
        if len(set(prefixes.values())) > 1:
            a, b = list(prefixes)[:2]
            first = next(
                (i for i in range(m) if prefixes[a][i] != prefixes[b][i]), None
            )
            self._violate(
                ViolationKind.CONVERGENCE,
                f"replicas disagree at position {first} of their common "
                f"prefix ({a}={prefixes[a][first]}, {b}={prefixes[b][first]})",
                position=first, nodes=[a, b],
            )

    def check_converged(self):
        """The stronger claim, valid only after the cluster has quiesced."""
        digests = self.cluster.digests()
        if len(set(digests.values())) > 1:
            self._violate(
                ViolationKind.CONVERGENCE,
                f"replicas failed to converge after quiescing: "
                f"{ {k: v[:2] for k, v in digests.items()} }",
            )

    # 3 ------------------------------------------------------------------

    def _check_index_watermark(self):
        """A replica's index must hold exactly the inserts it claims applied.

        The hazard: index and watermark are two pieces of state that must move
        together. Keep the watermark and drop the index and a replica reports
        itself caught up while holding nothing -- silent, total data loss that
        looks healthy from outside.
        """
        for nid, r in self.cluster.replicas.items():
            if not r.alive:
                continue
            expected = sum(
                1 for i, e in r.observed.items()
                if i <= r.sm.last_applied_index and e.command.get("op") == "insert"
            )
            if r.sm.size != expected:
                self._violate(
                    ViolationKind.INDEX_WATERMARK,
                    f"{nid} holds {r.sm.size} vectors but claims to have applied "
                    f"{expected} inserts up to index {r.sm.last_applied_index}",
                    node=nid, held=r.sm.size, claimed=expected,
                )

    # 4 and 5 ------------------------------------------------------------

    def _check_phantoms_and_duplicates(self):
        for nid, r in self.cluster.replicas.items():
            if not r.alive:
                continue

            committed = {
                e.command["id"]: i
                for i, e in r.observed.items()
                if e.command.get("op") == "insert"
            }
            order = r.sm.ids_in_order

            if len(order) != len(set(order)):
                dupes = [x for x in set(order) if order.count(x) > 1]
                self._violate(
                    ViolationKind.EXACTLY_ONCE,
                    f"{nid} applied {dupes[:3]} more than once",
                    node=nid, ids=dupes[:5],
                )

            for eid in order:
                if eid not in committed:
                    self._violate(
                        ViolationKind.PHANTOM,
                        f"{nid} holds {eid} with no committed log entry for it",
                        node=nid, id=eid,
                    )
                    break
                idx = committed[eid]
                seen = self._id_index.get(eid)
                if seen is not None and seen != idx:
                    self._violate(
                        ViolationKind.EXACTLY_ONCE,
                        f"{eid} was committed at two different log indices "
                        f"({seen} and {idx})",
                        id=eid, indices=[seen, idx],
                    )
                self._id_index[eid] = idx

    # 6 ------------------------------------------------------------------

    def check_read_honesty(self, before_acked: set[str], served_by: str,
                           read_index: int, hits: set[str] | None = None) -> None:
        """A linearizable read must be served from state containing every write
        acked before it began.

        CONTAINMENT, NOT SEARCH RECALL. An earlier version of this asserted
        that a k-NN query returned every acked id, and the fuzz sweep reported
        a violation on seed 57. Replaying it showed all 8 acked ids present in
        the serving replica's index while an ANN query at k=13 returned only 6
        of them.

        That was a flaw here, not in the platform. Two different properties had
        been conflated:

          STALENESS (consistency) -- is the state being read from fresh? That
            is what ReadIndex guarantees and what this invariant is about.
          RECALL (approximation)  -- does a k-NN query enumerate everything?
            HNSW is an APPROXIMATE index and never promised this. Its own
            recall figure is recall@10 against the true 10 nearest, not
            "returns all N when asked for N".

        Asserting recall here would report violations against a perfectly
        correct system, and worse, would make the read-consistency claim
        untestable by burying it in an unrelated property. Search recall is
        still recorded below, as an observation rather than a failure.
        """
        replica = self.cluster.replicas.get(served_by)
        if replica is None:
            return
        missing = {e for e in before_acked if not replica.sm.contains(e)}
        if missing:
            self._violate(
                ViolationKind.READ_HONESTY,
                f"a linearizable read at index {read_index} was served from "
                f"{served_by}, whose state is missing {len(missing)} "
                f"embedding(s) acked before the read began, e.g. "
                f"{sorted(missing)[:3]}",
                read_index=read_index, served_by=served_by,
                missing=sorted(missing)[:5],
            )

        if hits is not None and before_acked:
            found = len(before_acked & hits)
            self.result.read_recall_samples.append(
                {"acked": len(before_acked), "returned_by_search": found,
                 "recall": round(found / len(before_acked), 4)}
            )

    # ------------------------------------------------- non-violation tallies

    def note_stale_lag(self) -> None:
        self.result.stale_reads_lagging += 1

    def note_minority_refusal(self) -> None:
        self.result.minority_reads_refused += 1

    def note_uncommitted_loss(self, n: int = 1) -> None:
        self.result.uncommitted_writes_lost += n
