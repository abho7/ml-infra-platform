"""
The control plane: one object that owns the whole platform's state.

Everything above this -- the HTTP API, the CLI, the dashboard -- is a view onto
this class. It owns the storage cluster, the training jobs, and the single
clock they share, and it is the only place that knows how to advance them
together.

WHY ONE CLOCK. The storage cluster ticks (Raft elections, heartbeats, message
delivery) and training jobs step (gradient rounds). Those are different units
of work, and if each layer advanced on its own schedule then "a storage node
was partitioned while a training worker died" would have no well-defined
meaning -- the two events could not be ordered. So the plane drives both from
one `advance()`, and every event carries the storage tick. That is what makes
the cross-layer timeline in the dashboard real rather than a rendering
convenience.

JOBS ARE STEPPED, NOT RUN. A job never runs to completion inside a call here.
The plane advances it one step at a time, which is what allows a fault to be
scheduled *at* training step 37 rather than merely before or after the job. It
is also what lets the dashboard show a job progressing rather than jumping from
queued to done.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from mlplat.observe.events import EventKind, EventLog
from mlplat.storage.client import StorageClient
from mlplat.storage.cluster import StorageCluster
from mlplat.storage.readindex import ReadMode, ReadRefused
from mlplat.training.job import JobSpec, TrainingJob
from mlplat.training.publisher import EmbeddingPublisher


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PUBLISHING = "publishing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class ManagedJob:
    spec: JobSpec
    job: TrainingJob
    publisher: EmbeddingPublisher
    state: JobState = JobState.QUEUED
    started_tick: int | None = None
    finished_tick: int | None = None
    publish_reports: list = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> dict:
        r = self.job.result()
        return {
            "job_id": self.spec.job_id,
            "state": self.state.value,
            "step": self.job.step,
            "total_steps": self.spec.steps,
            "progress": round(self.job.step / max(1, self.spec.steps), 4),
            "workers": len(self.job.engine.live_ids),
            "workers_lost": list(self.job.workers_lost),
            "strategy": self.spec.strategy,
            "loss": r.loss_curve[-1] if r.loss_curve else None,
            "accuracy": r.final_accuracy if self.state is JobState.DONE else None,
            "embeddings_emitted": self.job.embeddings_emitted,
            "published": len(self.publisher.published),
            "pending": len(self.publisher.pending),
            "rejected": len(self.publisher.rejected),
            "started_tick": self.started_tick,
            "finished_tick": self.finished_tick,
            "error": self.error,
        }


class ControlPlane:
    def __init__(
        self,
        *,
        node_ids: list[str] | None = None,
        dim: int = 8,
        seed: int = 7,
        events: EventLog | None = None,
    ) -> None:
        self.events = EventLog() if events is None else events
        self.dim = dim
        self.node_ids = node_ids or [f"n{i + 1}" for i in range(5)]
        self.cluster = StorageCluster(
            self.node_ids, dim=dim, seed=seed, events=self.events
        )
        self.client = StorageClient(self.cluster)
        self.jobs: dict[str, ManagedJob] = {}
        self.started_at = time.time()

        self.cluster.run_until_leader()
        self.events.record(
            self.cluster.tick, EventKind.CLUSTER_STARTED,
            nodes=self.node_ids, dim=dim, leader=self.cluster.leader(),
        )

    # ------------------------------------------------------------------ jobs

    def submit(self, spec: JobSpec) -> ManagedJob:
        if spec.job_id in self.jobs:
            raise ValueError(f"job {spec.job_id!r} already exists")
        if spec.embedding_dim != self.dim:
            # Caught here rather than at the first write, where it would
            # surface as an opaque shape error from inside HNSW after the
            # cluster had already accepted the entry into its log.
            raise ValueError(
                f"job {spec.job_id!r} produces {spec.embedding_dim}-dim "
                f"embeddings but the cluster stores {self.dim}-dim"
            )

        # One clock for both layers: job events are stamped with the storage
        # tick, which is the only quantity that orders across them.
        job = TrainingJob(spec, events=self.events,
                          clock=lambda: self.cluster.tick)
        managed = ManagedJob(
            spec=spec, job=job,
            publisher=EmbeddingPublisher(self.client, events=self.events),
        )
        self.jobs[spec.job_id] = managed
        self.events.record(
            self.cluster.tick, EventKind.JOB_QUEUED,
            job=spec.job_id, steps=spec.steps, workers=spec.n_workers,
        )
        return managed

    def job(self, job_id: str) -> ManagedJob:
        if job_id not in self.jobs:
            raise KeyError(f"no such job: {job_id!r}")
        return self.jobs[job_id]

    # --------------------------------------------------------------- driving

    def advance(self, *, storage_ticks: int = 1, job_steps: int = 1) -> dict:
        """Advance the whole platform by one unit of work.

        Storage first, then training. The order matters and is not arbitrary: a
        job publishing this round should see a cluster that has already
        processed the consequences of anything that happened last round -- an
        election in progress, a partition just healed. Advancing training first
        would have jobs writing into a cluster whose state is one tick stale.
        """
        self.cluster.run(storage_ticks)

        progressed = []
        for managed in self.jobs.values():
            if managed.state in (JobState.DONE, JobState.FAILED):
                continue
            if managed.state is JobState.QUEUED:
                managed.state = JobState.RUNNING
                managed.started_tick = self.cluster.tick

            for _ in range(job_steps):
                if managed.job.finished:
                    break
                try:
                    ck = managed.job.run_step()
                except Exception as e:  # pragma: no cover - defensive
                    managed.state = JobState.FAILED
                    managed.error = f"{type(e).__name__}: {e}"
                    break
                if ck is not None:
                    managed.state = JobState.PUBLISHING
                    managed.publish_reports.append(managed.publisher.publish(ck))
                    managed.state = JobState.RUNNING
                progressed.append(managed.spec.job_id)

            if managed.job.finished and managed.state is not JobState.FAILED:
                managed.publisher.resolve_pending()
                managed.state = JobState.DONE
                managed.finished_tick = self.cluster.tick

        return {"tick": self.cluster.tick, "progressed": sorted(set(progressed))}

    def run_until_idle(self, *, max_rounds: int = 5000) -> int:
        rounds = 0
        while any(
            m.state not in (JobState.DONE, JobState.FAILED) for m in self.jobs.values()
        ):
            self.advance()
            rounds += 1
            if rounds >= max_rounds:
                break
        return rounds

    # ------------------------------------------------------------ membership

    def kill_node(self, node_id: str) -> dict:
        lost = self.cluster.kill(node_id)
        return {"node": node_id, "alive": False, "inflight_lost": lost}

    def restart_node(self, node_id: str) -> dict:
        self.cluster.restart(node_id)
        return {"node": node_id, "alive": True}

    def partition(self, *groups) -> dict:
        """Varargs, matching StorageCluster.partition.

        These two had different signatures -- a list-of-lists here, varargs
        below -- which is the sort of inconsistency that only shows up as a
        TypeError from a caller who reasonably assumed they matched. The HTTP
        layer unpacks its JSON list at the boundary instead.
        """
        self.cluster.partition(*[set(g) for g in groups])
        return {"partition": [sorted(g) for g in groups]}

    def heal(self) -> dict:
        self.cluster.heal()
        return {"partition": None}

    def kill_worker(self, job_id: str, worker_id: str | None = None) -> dict:
        """Kill a training worker. Omit `worker_id` to kill any live one.

        The dashboard and the session recorder both want "lose a worker" without
        caring which, and having every caller reach into `engine.live_ids` to
        pick one itself is how that logic ends up duplicated three ways with
        three different empty-list behaviours.

        The last live worker is NOT killable here. The training engine's
        re-sharding has nothing to re-shard onto, and a job with zero workers is
        a wedged job, not a fault-tolerance result.
        """
        managed = self.job(job_id)
        live = managed.job.engine.live_ids
        if worker_id in (None, ""):
            if len(live) <= 1:
                raise ValueError(
                    f"job {job_id!r} has {len(live)} live worker(s); "
                    "killing the last one wedges the job rather than testing it"
                )
            worker_id = live[-1]
        managed.job.kill_worker(worker_id)
        return {
            "job": job_id, "worker": worker_id,
            "survivors": managed.job.engine.live_ids,
        }

    # ---------------------------------------------------------------- queries

    def search(self, vector, k: int = 5, *, mode: str = "linearizable") -> dict:
        try:
            res = self.client.search(
                np.asarray(vector, dtype=np.float64), k, mode=ReadMode(mode)
            )
        except ReadRefused as e:
            # Surfaced as a result rather than an exception: an API caller
            # needs to distinguish "the cluster refused to serve this safely"
            # from "the server broke", and those are very different things.
            return {"ok": False, "mode": mode, "reason": str(e), "hits": []}
        return {
            "ok": True, "mode": res.mode, "served_by": res.served_by,
            "read_index": res.read_index, "ticks": res.ticks,
            "hits": [
                {"id": h["id"], "distance": round(float(h["distance"]), 6),
                 "metadata": h["metadata"]}
                for h in res.hits
            ],
        }

    # ------------------------------------------------------------------ state

    def state(self) -> dict:
        """Everything the dashboard and CLI need, in one call."""
        cluster = self.cluster.state()
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "tick": self.cluster.tick,
            "cluster": cluster,
            "health": self._health(cluster),
            "jobs": [m.to_json() for m in self.jobs.values()],
            "storage": {
                "acked_writes": len(self.client.acked),
                "reads": self.client.reader.stats(),
            },
            "events": len(self.events),
        }

    def _health(self, cluster: dict) -> dict:
        """A single honest summary of whether the cluster can do its job.

        `quorum` is the load-bearing field: a cluster with a leader but no
        quorum reachable cannot commit, and reporting it as healthy because a
        leader exists is exactly the kind of dashboard that lies.
        """
        live = [n for n, s in cluster["nodes"].items() if s["alive"]]
        total = len(cluster["nodes"])
        needed = total // 2 + 1
        partitioned = cluster["partition"] is not None

        reachable = len(live)
        if partitioned:
            groups = cluster["partition"]
            reachable = max(
                (len([n for n in g if cluster["nodes"][n]["alive"]]) for g in groups),
                default=0,
            )

        has_quorum = reachable >= needed
        if has_quorum and not partitioned and len(live) == total:
            status = "healthy"
        elif has_quorum:
            status = "degraded"
        else:
            status = "unavailable"

        return {
            "status": status,
            "live_nodes": len(live),
            "total_nodes": total,
            "quorum_needed": needed,
            "largest_reachable_group": reachable,
            "has_quorum": has_quorum,
            "partitioned": partitioned,
            "leader": cluster["leader"],
        }

    def recent_events(self, limit: int = 50) -> list[dict]:
        return [e.to_json() for e in self.events.events[-limit:]]
