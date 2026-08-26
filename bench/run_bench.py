"""
End-to-end benchmarks for the whole pipeline.

Every number here is measured from a run that actually executed. Where a figure
is in simulated ticks rather than seconds, it says so and says why: the storage
cluster advances on an explicit tick, so "how long did a write take" has two
honest answers -- how many consensus rounds it needed (a property of the
protocol, portable to any deployment) and how many microseconds of CPU that
cost here (a property of this laptop). Both are reported, never mixed.

Four questions:

  1 END-TO-END LATENCY   from a training checkpoint existing to its embeddings
                         being queryable through a linearizable read
  2 SCALING              what happens as storage nodes and training workers are
                         added
  3 RECOVERY             how long the cluster takes to become writable again
                         after each kind of failure
  4 READ COST            what linearizable reads cost against stale ones

Question 4 is the one that justifies having implemented ReadIndex at all: if
the safe mode were free, nobody would need to be told which one they were
using.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from mlplat.control.plane import ControlPlane, JobState
from mlplat.observe.events import EventKind
from mlplat.storage.client import StorageClient
from mlplat.storage.cluster import StorageCluster
from mlplat.storage.readindex import ReadMode, ReadRefused
from mlplat.training.job import JobSpec

DIM = 8


def _plane(nodes=5, dim=DIM, seed=7):
    p = ControlPlane(node_ids=[f"n{i + 1}" for i in range(nodes)], dim=dim, seed=seed)
    return p


# ------------------------------------------------------- 1. end-to-end latency


def end_to_end_latency(*, nodes=5, workers=4, steps=40, repeats=3) -> dict:
    """Checkpoint produced -> embeddings queryable.

    Measured on the real path: the job publishes through consensus and the
    result is confirmed with a linearizable read, so the number includes the
    ReadIndex round trip a real client would pay.
    """
    samples = []
    for rep in range(repeats):
        p = _plane(nodes, seed=7 + rep)
        p.submit(JobSpec(job_id="j", n_workers=workers, steps=steps,
                         n_classes=DIM, publish_every=10, seed=13 + rep))
        managed = p.job("j")

        while managed.state is not JobState.DONE:
            before_pub = len(managed.publisher.published)
            t0 = time.perf_counter()
            tick0 = p.cluster.tick
            p.advance()
            if len(managed.publisher.published) > before_pub:
                # A checkpoint just landed. Confirm it is actually readable.
                eid = sorted(managed.publisher.published)[-1]
                sm = p.cluster.replicas[p.cluster.leader()].sm
                vec = sm._db._index.vectors[sm._db._id_map[eid]]
                try:
                    res = p.client.search(np.asarray(vec), k=1,
                                          mode=ReadMode.LINEARIZABLE)
                    ok = res.hits and res.hits[0]["id"] == eid
                except ReadRefused:
                    ok = False
                samples.append({
                    "wall_ms": (time.perf_counter() - t0) * 1000,
                    "ticks": p.cluster.tick - tick0,
                    "embeddings": len(managed.publisher.published) - before_pub,
                    "queryable": bool(ok),
                })

    walls = sorted(s["wall_ms"] for s in samples)
    ticks = sorted(s["ticks"] for s in samples)
    return {
        "kind": "measured",
        "checkpoints": len(samples),
        "all_queryable": all(s["queryable"] for s in samples),
        "wall_ms": {
            "p50": round(walls[len(walls) // 2], 2),
            "p95": round(walls[int(len(walls) * 0.95)], 2),
            "max": round(walls[-1], 2),
        },
        "consensus_ticks": {
            "p50": ticks[len(ticks) // 2],
            "p95": ticks[int(len(ticks) * 0.95)],
            "max": ticks[-1],
        },
        "note": (
            "wall_ms is this machine; consensus_ticks is the protocol cost and "
            "is what would carry to a real deployment. Each sample covers one "
            "checkpoint of embeddings plus a linearizable read confirming them."
        ),
    }


# --------------------------------------------------------------- 2. scaling


def scaling(node_counts=(3, 5, 7, 9), worker_counts=(2, 4, 8), steps=30) -> dict:
    """Two independent sweeps: storage width and training width.

    Held separate on purpose. Adding storage nodes makes writes MORE expensive
    (a larger quorum to convince) while adding training workers makes steps
    cheaper (a smaller shard each). Sweeping them together would net the two
    effects against each other and report a flat line that means nothing.
    """
    storage_rows = []
    for n in node_counts:
        p = _plane(n, seed=7)
        p.submit(JobSpec(job_id="j", n_workers=4, steps=steps,
                         n_classes=DIM, publish_every=10))
        t0 = time.perf_counter()
        rounds = p.run_until_idle()
        wall = time.perf_counter() - t0
        m = p.job("j")
        storage_rows.append({
            "kind": "measured", "nodes": n, "workers": 4,
            "rounds": rounds, "ticks": p.cluster.tick,
            "wall_s": round(wall, 3),
            "published": len(m.publisher.published),
            "ticks_per_write": round(
                p.cluster.tick / max(1, len(m.publisher.published)), 2),
            "quorum": n // 2 + 1,
        })

    worker_rows = []
    for w in worker_counts:
        p = _plane(5, seed=7)
        p.submit(JobSpec(job_id="j", n_workers=w, steps=steps,
                         n_classes=DIM, publish_every=10))
        t0 = time.perf_counter()
        rounds = p.run_until_idle()
        wall = time.perf_counter() - t0
        m = p.job("j")
        worker_rows.append({
            "kind": "measured", "nodes": 5, "workers": w,
            "rounds": rounds, "wall_s": round(wall, 3),
            "final_accuracy": round(m.job.result().final_accuracy, 4),
            "published": len(m.publisher.published),
        })

    return {"storage": storage_rows, "training": worker_rows,
            "note": ("Storage and training width are swept separately: more "
                     "nodes make a write costlier, more workers make a step "
                     "cheaper, and sweeping together would cancel them out.")}


# -------------------------------------------------------------- 3. recovery


COMMIT_TIMEOUT = 25


def recovery(repeats=3) -> dict:
    """Time to become writable again after each failure kind.

    Measured as ticks from the fault firing to the first write that commits
    afterwards -- which is what a client actually experiences, rather than time
    to elect a leader. A cluster with a fresh leader that cannot yet accept
    writes has not recovered.
    """
    def time_to_writable(setup, label):
        out = []
        for rep in range(repeats):
            c = StorageCluster([f"n{i + 1}" for i in range(5)],
                               dim=DIM, seed=7 + rep)
            c.run_until_leader()
            # A SHORT commit timeout, deliberately. With the default 200,
            # a write aimed at a soon-to-be-deposed leader burns the full
            # timeout before the client gives up and retries, and that
            # patience -- not the cluster -- dominates the result. It made
            # "leader isolated" read 283 ticks against "leader killed" at 15,
            # a gap that was almost entirely the timeout path. At 25 the
            # numbers compare like for like and measure the cluster.
            client = StorageClient(c, commit_timeout=COMMIT_TIMEOUT)
            for i in range(5):
                client.write(f"pre{i}", np.random.default_rng(i).normal(size=DIM), {})

            fault_tick = c.tick
            setup(c)

            # Retry with a FRESH ID each attempt, never the same one.
            #
            # This measures when the CLUSTER became writable again, which is
            # what the row claims. A single write cannot measure it: the client
            # deliberately refuses to re-propose an appended-but-unconfirmed
            # entry (see client.py -- re-proposing is what caused this
            # project's duplicate-commit bug), so a write that lands on a
            # soon-to-be-deposed leader returns `pending` and stops. An earlier
            # version of this benchmark did exactly that and reported "leader
            # isolated: 0/3 recovered", which said nothing about the cluster
            # and everything about one ambiguous entry.
            #
            # A fresh id per attempt is also what a real client does: retrying
            # safely means a new idempotency key, not the same command again.
            committed = False
            for attempt in range(8):
                r = client.write(f"post-{label}-{rep}-{attempt}",
                                 np.ones(DIM), {}, retries=1)
                if r.committed:
                    committed = True
                    break
                c.run_until_leader(max_ticks=80)
            out.append({
                "recovered": committed,
                "ticks": (c.tick - fault_tick) if committed else None,
            })
        ok = [o for o in out if o["recovered"]]
        return {
            "kind": "measured", "fault": label,
            "runs": len(out), "recovered": len(ok),
            "median_ticks": (sorted(o["ticks"] for o in ok)[len(ok) // 2]
                             if ok else None),
            "max_ticks": max((o["ticks"] for o in ok), default=None),
        }

    def kill_follower(c):
        c.kill(next(n for n in c.live_ids if n != c.leader()))

    def kill_leader(c):
        c.kill(c.leader())

    def partition_minority(c):
        lead = c.leader()
        others = [n for n in c.node_ids if n != lead][:1]
        c.partition(set(others), set(c.node_ids) - set(others))

    def isolate_leader(c):
        lead = c.leader()
        c.partition({lead}, set(c.node_ids) - {lead})

    def kill_two(c):
        lead = c.leader()
        c.kill(lead)
        c.kill(next(n for n in c.live_ids))

    return {
        "rows": [
            time_to_writable(kill_follower, "follower killed"),
            time_to_writable(kill_leader, "leader killed"),
            time_to_writable(partition_minority, "one node partitioned off"),
            time_to_writable(isolate_leader, "leader isolated"),
            time_to_writable(kill_two, "leader + one follower killed"),
        ],
        "commit_timeout_ticks": COMMIT_TIMEOUT,
        "note": ("Ticks from the fault to the first write that COMMITS after "
                 "it, retrying with a fresh id each attempt. Time to elect a "
                 "leader would be smaller and less honest: a cluster with a "
                 "leader that cannot yet accept writes has not recovered. The "
                 f"client's commit timeout is {COMMIT_TIMEOUT} ticks and is "
                 "part of the number -- at the default 200 the client's own "
                 "patience dominated and the rows stopped being comparable."),
    }


# ------------------------------------------------------------- 4. read cost


def read_cost(n_reads=40) -> dict:
    """What the safe read mode costs against the fast one."""
    p = _plane(5, seed=7)
    p.submit(JobSpec(job_id="j", steps=30, n_classes=DIM, publish_every=10))
    p.run_until_idle()

    sm = p.cluster.replicas[p.cluster.leader()].sm
    ids = sm.ids_in_order[:n_reads]
    vecs = [sm._db._index.vectors[sm._db._id_map[i]] for i in ids]

    lin_ticks, lin_wall = [], []
    for v in vecs:
        t0 = time.perf_counter()
        t = p.cluster.tick
        p.client.search(np.asarray(v), k=5, mode=ReadMode.LINEARIZABLE)
        lin_ticks.append(p.cluster.tick - t)
        lin_wall.append((time.perf_counter() - t0) * 1000)

    stale_ticks, stale_wall = [], []
    for v in vecs:
        t0 = time.perf_counter()
        t = p.cluster.tick
        p.client.search(np.asarray(v), k=5, mode=ReadMode.STALE)
        stale_ticks.append(p.cluster.tick - t)
        stale_wall.append((time.perf_counter() - t0) * 1000)

    med = lambda xs: sorted(xs)[len(xs) // 2]
    return {
        "kind": "measured",
        "reads": n_reads,
        "linearizable": {"median_ticks": med(lin_ticks),
                         "median_ms": round(med(lin_wall), 3)},
        "stale": {"median_ticks": med(stale_ticks),
                  "median_ms": round(med(stale_wall), 3)},
        "tick_overhead": med(lin_ticks) - med(stale_ticks),
        "note": ("A linearizable read pays a heartbeat quorum round to confirm "
                 "leadership; a stale read pays nothing and may lag. There is "
                 "no leader lease here, so every linearizable read pays the "
                 "full round trip -- a real system amortises that."),
    }


def run_all() -> dict:
    return {
        "end_to_end_latency": end_to_end_latency(),
        "scaling": scaling(),
        "recovery": recovery(),
        "read_cost": read_cost(),
    }


if __name__ == "__main__":
    out = run_all()

    e = out["end_to_end_latency"]
    print("END-TO-END: checkpoint -> queryable")
    print(f"  {e['checkpoints']} checkpoints, all queryable: {e['all_queryable']}")
    print(f"  wall  p50={e['wall_ms']['p50']}ms p95={e['wall_ms']['p95']}ms")
    print(f"  ticks p50={e['consensus_ticks']['p50']} p95={e['consensus_ticks']['p95']}")

    print("\nSCALING (storage width)")
    print(f"  {'nodes':>6}{'quorum':>8}{'ticks':>8}{'wall_s':>9}{'ticks/write':>13}")
    for r in out["scaling"]["storage"]:
        print(f"  {r['nodes']:>6}{r['quorum']:>8}{r['ticks']:>8}"
              f"{r['wall_s']:>9}{r['ticks_per_write']:>13}")

    print("\nSCALING (training width)")
    print(f"  {'workers':>8}{'rounds':>8}{'wall_s':>9}{'accuracy':>10}")
    for r in out["scaling"]["training"]:
        print(f"  {r['workers']:>8}{r['rounds']:>8}{r['wall_s']:>9}"
              f"{r['final_accuracy']:>10}")

    print("\nRECOVERY (ticks to first committed write after the fault)")
    for r in out["recovery"]["rows"]:
        print(f"  {r['fault']:<30}{r['recovered']}/{r['runs']} recovered  "
              f"median={r['median_ticks']} max={r['max_ticks']}")

    rc = out["read_cost"]
    print(f"\nREAD COST over {rc['reads']} reads")
    print(f"  linearizable: {rc['linearizable']['median_ticks']} ticks, "
          f"{rc['linearizable']['median_ms']} ms")
    print(f"  stale:        {rc['stale']['median_ticks']} ticks, "
          f"{rc['stale']['median_ms']} ms")
    print(f"  overhead:     {rc['tick_overhead']} ticks")
