"""
A training job, driven on the unmodified training engine.

The engine already owns everything about *how* training works: worker sharding,
gradient reduction, straggler fencing, its six invariants. This wraps it and
answers a question the engine never had to: what does a trained model hand to
the storage layer, and when.

WHAT AN EMBEDDING IS HERE, precisely. The engine trains softmax regression, so
the model is a weight matrix W and a bias b. For an input x, the model's
pre-softmax output `logits(theta, x) = xW + b` is a learned representation of x
in class space -- it is literally the model's opinion about x, and it is the
only genuine learned output this model produces. That is what gets stored.

The dimension is therefore `n_classes`, so a job that wants richer embeddings
configures more classes. Nothing here invents a representation the model does
not actually compute: an alternative would have been to store raw features,
which would make the storage layer a database of the *inputs* and would have
nothing to do with training having happened.

CHECKPOINT PUBLISHING. Embeddings are emitted periodically during training, not
only at the end. That is what makes the cross-layer failure interesting: a
storage node can die while a training job is mid-flight and still has more
writes to come, which is a state that simply does not exist if publication is a
single event at the end.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

import mlplat.enginepath  # noqa: F401
from dtf.allreduce import RingAllReduce
from dtf.coordinator import Coordinator
from dtf.data import make_blobs
from dtf.model import accuracy, init_weights, logits
from dtf.paramserver import ParameterServer
from engines.sim import SimEngine

from mlplat.observe.events import EventKind, EventLog

STRATEGIES = {"all-reduce": RingAllReduce, "parameter-server": ParameterServer}


@dataclass
class JobSpec:
    job_id: str
    n_workers: int = 4
    steps: int = 60
    n_samples: int = 803
    n_features: int = 8
    n_classes: int = 8
    batch_size: int = 16
    lr: float = 0.5
    strategy: str = "all-reduce"
    seed: int = 13
    # How often to publish. A checkpoint every `publish_every` steps, plus a
    # final one, so a job produces several rounds of writes rather than one.
    publish_every: int = 20
    # How many sample embeddings each checkpoint emits. Bounded because the
    # storage layer is a real Raft cluster: every embedding is a consensus
    # round, and publishing all 803 every checkpoint would make the job's cost
    # entirely storage-bound and the training half decorative.
    embeddings_per_checkpoint: int = 8

    @property
    def embedding_dim(self) -> int:
        return self.n_classes


@dataclass
class Checkpoint:
    step: int
    loss: float
    accuracy: float
    ids: list[str] = field(default_factory=list)
    vectors: list[np.ndarray] = field(default_factory=list)
    metadata: list[dict] = field(default_factory=list)


@dataclass
class JobResult:
    job_id: str
    spec: JobSpec
    steps_completed: int
    final_loss: float
    final_accuracy: float
    checkpoints: int
    embeddings_emitted: int
    workers_lost: list[str] = field(default_factory=list)
    loss_curve: list[float] = field(default_factory=list)


class TrainingJob:
    """Drives the engine's Coordinator one step at a time.

    Stepped rather than run to completion in one call, so the orchestrator can
    interleave it with storage faults -- "kill a storage node at training step
    37" is only expressible if something outside owns the step loop.
    """

    def __init__(self, spec: JobSpec, *, events: EventLog | None = None,
                 clock: Callable[[], int] | None = None) -> None:
        self.spec = spec
        # THE SHARED CLOCK. Without this, job events were stamped with
        # `self.step` while storage events carried the cluster tick, and both
        # rendered in the same column on one timeline labelled "both layers,
        # one clock". They were not one clock: a job_step at 8 and a
        # leader_elected at 14 looked orderable and were not comparable at all.
        #
        # The job cannot read the cluster tick itself -- it has no reference to
        # a cluster, deliberately -- so whoever owns both layers supplies the
        # clock. The step number is still recorded, in `detail`, where it is a
        # fact about the job rather than a position on a shared timeline.
        self._clock = clock if clock is not None else (lambda: self.step)
        # `is None`, not `or`. EventLog defines __len__, so a fresh empty log is
        # FALSY -- `events or EventLog()` therefore discarded the shared log the
        # caller passed in and silently created a private one. Every layer got
        # its own timeline, which is precisely the opposite of this class's
        # purpose, and nothing failed until a test inspected the shared log.
        self.events = EventLog() if events is None else events

        self.dataset = make_blobs(
            n_samples=spec.n_samples,
            n_features=spec.n_features,
            n_classes=spec.n_classes,
            seed=spec.seed,
        )
        theta0 = init_weights(self.dataset.spec, seed=spec.seed + 1)
        self.worker_ids = [f"{spec.job_id}-w{i + 1}" for i in range(spec.n_workers)]

        self.engine = SimEngine(
            self.dataset,
            self.worker_ids,
            theta0=theta0,
            batch_size=spec.batch_size,
            seed=spec.seed + 2,
        )
        self.coord = Coordinator(
            self.engine,
            STRATEGIES[spec.strategy](self.worker_ids),
            lr=spec.lr,
            straggler_timeout=2,
        )

        self.step = 0
        self.loss_curve: list[float] = []
        self.workers_lost: list[str] = []
        self.embeddings_emitted = 0
        self.checkpoints_taken = 0
        self.finished = False

        # Which sample each emitted embedding came from. A running offset
        # rather than a fresh random draw per checkpoint, so ids are unique
        # across checkpoints and every embedding traces back to a specific
        # sample and a specific step.
        self._sample_cursor = 0

        self.events.record(
            self._clock(), EventKind.JOB_STARTED,
            job=spec.job_id, workers=spec.n_workers,
            strategy=spec.strategy, steps=spec.steps,
        )

    # ------------------------------------------------------------------ faults

    def kill_worker(self, worker_id: str) -> None:
        """Kill a training worker. The engine handles the consequences -- it
        re-shards across the survivors and shrinks the effective batch."""
        if worker_id not in self.engine.workers:
            raise KeyError(f"no such worker: {worker_id}")
        if not self.engine.workers[worker_id].alive:
            return
        self.engine.kill(worker_id)
        self.workers_lost.append(worker_id)
        if self.engine.live_ids:
            self.engine.reassign_shards()
            self.coord.strategy = self.coord.strategy.with_members(self.engine.live_ids)
        self.events.record(
            self._clock(), EventKind.WORKER_DIED,
            job=self.spec.job_id, worker=worker_id,
            survivors=len(self.engine.live_ids),
        )

    # ------------------------------------------------------------------- step

    def run_step(self) -> Checkpoint | None:
        """One training step. Returns a Checkpoint when one is due.

        Returning the checkpoint rather than writing it is deliberate: this
        class knows how to train, and knows nothing about storage. The
        publisher owns the write path, so a job cannot accidentally bypass
        consensus.
        """
        if self.finished:
            return None
        if not self.engine.live_ids:
            self.finished = True
            self.events.record(
                self._clock(), EventKind.JOB_FINISHED,
                job=self.spec.job_id, reason="every worker died",
            )
            return None

        rec = self.coord.run_step()
        self.loss_curve.append(rec.mean_loss)
        self.step += 1
        self.engine.advance()

        self.events.record(
            self._clock(), EventKind.JOB_STEP,
            job=self.spec.job_id, step=rec.step,
            loss=rec.mean_loss, effective_batch=rec.effective_batch,
            contributors=len(rec.contributors),
        )

        due = self.step % self.spec.publish_every == 0
        last = self.step >= self.spec.steps
        if last:
            self.finished = True
            self.events.record(
                self._clock(), EventKind.JOB_FINISHED,
                job=self.spec.job_id, steps=self.step,
                final_loss=rec.mean_loss,
            )
        if due or last:
            return self._checkpoint()
        return None

    # ------------------------------------------------------------- embeddings

    def _checkpoint(self) -> Checkpoint:
        """Embeddings for the next slice of samples, from the CURRENT weights.

        Weights are read from the coordinator, which holds the authoritative
        post-reduction parameters -- not from a worker, whose copy is only
        guaranteed correct at a barrier.
        """
        spec = self.spec
        n = spec.embeddings_per_checkpoint
        idx = [
            (self._sample_cursor + i) % len(self.dataset)
            for i in range(n)
        ]
        self._sample_cursor += n

        X = self.dataset.X[idx]
        y = self.dataset.y[idx]
        emb = logits(self.coord.theta, X, self.dataset.spec)

        acc = accuracy(self.coord.theta, self.dataset.X, self.dataset.y, self.dataset.spec)
        ck = Checkpoint(
            step=self.step,
            loss=self.loss_curve[-1] if self.loss_curve else float("nan"),
            accuracy=acc,
        )
        for j, sample_i in enumerate(idx):
            ck.ids.append(f"{spec.job_id}:s{sample_i}@{self.step}")
            ck.vectors.append(np.asarray(emb[j], dtype=np.float64))
            ck.metadata.append({
                "job": spec.job_id,
                "step": self.step,
                "sample": int(sample_i),
                "label": int(y[j]),
                "accuracy_at_step": acc,
            })

        self.checkpoints_taken += 1
        self.embeddings_emitted += len(ck.ids)
        return ck

    # ----------------------------------------------------------------- result

    def result(self) -> JobResult:
        return JobResult(
            job_id=self.spec.job_id,
            spec=self.spec,
            steps_completed=self.step,
            final_loss=self.loss_curve[-1] if self.loss_curve else float("nan"),
            final_accuracy=accuracy(
                self.coord.theta, self.dataset.X, self.dataset.y, self.dataset.spec
            ),
            checkpoints=self.checkpoints_taken,
            embeddings_emitted=self.embeddings_emitted,
            workers_lost=list(self.workers_lost),
            loss_curve=[float(x) for x in self.loss_curve],
        )
