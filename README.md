# ml-infra-platform

[![tests](https://github.com/abho7/ml-infra-platform/actions/workflows/tests.yml/badge.svg)](https://github.com/abho7/ml-infra-platform/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A distributed ML platform built by composing three systems that were already
built and tested separately: a Raft consensus engine, an HNSW vector index, and
a distributed training framework. Models train on the training engine; their
embeddings are replicated through Raft into an HNSW-backed vector store that
survives node failure; a control plane orchestrates both and a dashboard shows
it happening.

**[Report and benchmarks](https://abho7.github.io/ml-infra-platform/)** ·
**[Dashboard replay](https://abho7.github.io/ml-infra-platform/dashboard.html)**

None of the three engines is modified. They are cloned in as read-only
dependencies and driven from outside.

---

## The point

Each engine passes its own suite: 23 tests for Raft, 28 for HNSW, 68 for the
training framework. None of those suites says anything about what happens when
a training worker dies at the same instant the storage cluster splits in half.

A component proof is a statement about a component under its own assumptions.
Composing components creates states that neither one's assumptions cover. The
first bug found here is exactly that shape: Raft behaved correctly, HNSW
behaved correctly, and the platform crashed every replica in the cluster.

So the claim this project actually makes is about the **composition**, and the
cross-layer chaos harness is where that claim is earned.

## Layout

```
mlplat/
  storage/     transport, cluster, state machine, ReadIndex, client
  training/    training job, embedding publisher
  control/     control plane, HTTP API, CLI
  chaos/       cross-layer faults, scenarios, fuzz sweep, invariant checker
  observe/     unified event log, session recorder
tests/         per-layer, composition, and real-TCP cross-validation
bench/         end-to-end latency, scaling, recovery, read cost
dashboard/     live UI; the same UI replays a recorded session
```

The three engines live in `raft-engine/`, `hnsw-engine/` and `training-engine/`.
`mlplat/enginepath.py` puts them on `sys.path`. Everything written here lives
under `mlplat.*` because the training engine already occupies the top-level
names `dtf`, `engines` and `chaos`.

## Running it

```bash
git clone https://github.com/abho7/raft-kv-store.git             ./raft-engine
git clone https://github.com/abho7/vectordb-hnsw.git             ./hnsw-engine
git clone https://github.com/abho7/distributed-training-framework.git ./training-engine
pip install numpy pytest

pytest -q                              # the whole suite
pytest -q -m "not slow"                # skip the long chaos and TCP tests
python -m mlplat.chaos.fuzz 60         # randomized cross-layer sweep
python -m mlplat.control.cli serve     # live control plane + dashboard :8950
python -m mlplat.control.cli demo      # a scripted run in the terminal
python run_experiments.py              # regenerate every number on the site
python build_site.py                   # rebuild the site from those results
```

Dependencies are numpy and pytest. The consensus engine, the vector index, the
training framework, the transport, the control plane and its HTTP API are all
from scratch.

## The consistency guarantee

| operation | guarantee | under partition |
|---|---|---|
| **write** | Linearizable. Acknowledged only after quorum commit in the leader's current term. | The minority side cannot commit. The write returns unacknowledged and is never claimed as durable. |
| **read: linearizable** | Reflects every write acknowledged before the read began. | **Refuses.** Leadership cannot be confirmed, so the read fails rather than serving state it cannot vouch for. |
| **read: stale** | Served from any replica's local index, no coordination. May lag arbitrarily. Opt-in, never the default. | Succeeds, possibly with old data. That is the mode's contract. |

Linearizable reads are **not** inherited from the storage engine, which serves
reads from whichever node receives them and names that as a known
simplification. They are implemented here as a ReadIndex protocol layered on
top: record the leader's commit index, confirm leadership with a heartbeat
quorum round, then wait until `last_applied >= read_index` before serving.

## Cross-layer invariants

The engines' own invariants are already proven. These six exist only at the
seams, and are checked after every round rather than once at the end:

1. **Acknowledged-write durability** — an embedding acked to a training job is
   present on every subsequently elected leader.
2. **Replica convergence** — any two in-sync replicas hold identical applied
   sequences. Catches seed divergence and apply-order drift.
3. **Index/watermark agreement** — a replica's index contains exactly the
   committed prefix it claims to have applied.
4. **No phantom embeddings** — nothing is searchable that was not committed
   through Raft.
5. **Exactly-once accounting** — no embedding stored twice, including across a
   restart that re-applies a log prefix.
6. **Linearizable-read honesty** — a read never returns state older than a write
   acknowledged before it began, and fails rather than lies.

Each has a mutation test that breaks it deliberately and asserts the checker
fires and names the right one. A correctness harness that has never been
observed to fail is not evidence of anything.

**Deliberately not violations:** a stale read lagging (that is the mode's
contract); a minority-side linearizable read failing (correct, and the point of
ReadIndex); an unacknowledged in-flight write vanishing when its node dies
(correct, and the analogue of an uncommitted Raft entry). The last is reported
separately as *lost uncommitted* so it stays visible rather than hidden.

## Bugs found in the composition

Three, all fixed, each with a regression test. Full write-ups on the report.

1. **The same embedding id committed at two log indices.** A write proposed,
   timed out unconfirmed, and re-proposed can commit twice. The watermark can't
   catch it — both indices are above it — so `VectorDB.insert` raised on the
   duplicate and killed apply on every replica at once. Neither engine was at
   fault; the defect was treating an unconfirmed proposal as safe to retry.
   Fixed on both sides: the client returns instead of re-proposing, and the
   state machine suppresses a repeated id, because a state machine over a
   replicated log must be **total**.

2. **A shared event log silently replaced by an empty one.** `EventLog` defines
   `__len__`, so an empty log is falsy, and `events or EventLog()` discarded the
   caller's log in three constructors — precisely when it was new and empty,
   which is always at startup.

3. **The correctness checker was testing the wrong property.** Invariant 6 once
   asserted that a k-NN query returns every acknowledged id. HNSW is an
   approximate index and a k-NN query is not an enumeration; the check conflated
   recall with staleness. The only violation the sweep ever reported was this,
   the harness's own bug.

## Verification

Numbers on the report come from `run_experiments.py` and nowhere else. The
build reads `site/results.json`; there is no literal figure in the page
template, and a missing value prints as "not measured" rather than falling back
to something plausible.

The deterministic transport is cross-validated against the Raft engine's real
asyncio TCP server: three servers on localhost with this platform's state
machine attached, driven through the engine's own client. Replicas converge to
identical digests, acknowledged writes survive a node dying, and reverting the
duplicate-id fix reproduces the original `ValueError` over TCP too — which is
how those tests are known not to be vacuous.

## Known limitations

- **No snapshots.** Neither the Raft engine nor this platform truncates the log,
  so a rejoining replica catches up by full replay and the log grows without
  bound. The first thing a real deployment would need.
- **No leader lease.** Every linearizable read pays a full heartbeat quorum
  round. A real system amortises that across a lease interval.
- **Deterministic transport, not a network.** Modelled latency, drops and
  partitions. That is what makes a fault reproducible from a seed, and it is
  also why the wall-clock numbers are not deployment numbers. See the
  cross-validation above for what this does and does not buy.
- **Deletion is soft.** Inherited from the vector engine; vectors are
  tombstoned, not reclaimed.
- **One shard.** Every replica holds the whole index, so storage capacity is one
  node's capacity. Sharding would be a replication-group-per-shard change.

---

Abhineeth Duddela · composed from
[raft-kv-store](https://github.com/abho7/raft-kv-store),
[vectordb-hnsw](https://github.com/abho7/vectordb-hnsw) and
[distributed-training-framework](https://github.com/abho7/distributed-training-framework),
none of them modified.
