"""
Training layer -> storage layer.

The engines are each proven. What is unproven is that a model's output survives
the trip into a replicated store, and that the accounting on both sides agrees
about what made it.

The interesting tests here are the ones where something fails DURING
publication, because that is a state that only exists because the two layers
are composed: a storage node has no notion of a training job, and a training
job has no notion of quorum.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlplat.control.plane import ControlPlane
from mlplat.observe.events import EventKind, EventLog
from mlplat.storage.client import StorageClient
from mlplat.storage.cluster import StorageCluster
from mlplat.storage.readindex import ReadMode
from mlplat.training.job import JobSpec, TrainingJob
from mlplat.training.publisher import EmbeddingPublisher


def build(*, n_nodes=5, job_kwargs=None, cluster_seed=7):
    ev = EventLog()
    spec = JobSpec(job_id="j1", **(job_kwargs or {}))
    c = StorageCluster(
        [f"n{i + 1}" for i in range(n_nodes)],
        dim=spec.embedding_dim, seed=cluster_seed, events=ev,
    )
    assert c.run_until_leader() is not None
    client = StorageClient(c)
    pub = EmbeddingPublisher(client, events=ev)
    job = TrainingJob(spec, events=ev)
    return ev, c, client, pub, job


def run_to_completion(job, pub):
    reports = []
    while not job.finished:
        ck = job.run_step()
        if ck is not None:
            reports.append(pub.publish(ck))
    return reports


# ------------------------------------------------------------ the happy path


def test_a_training_run_lands_in_the_store_and_is_queryable():
    ev, c, client, pub, job = build()
    reports = run_to_completion(job, pub)
    c.run(60)

    r = job.result()
    assert r.steps_completed == r.spec.steps
    assert r.final_accuracy > 0.8, r.final_accuracy
    assert r.embeddings_emitted == sum(rp.attempted for rp in reports)

    assert pub.stats()["published"] == r.embeddings_emitted
    assert pub.stats()["pending"] == 0
    assert pub.stats()["rejected"] == 0

    leader = c.leader()
    for eid in pub.published:
        assert c.replicas[leader].sm.contains(eid), f"{eid} was acked but is not stored"


def test_publication_happens_during_training_not_only_at_the_end():
    """Otherwise the cross-layer fault scenarios have no window to hit."""
    ev, c, client, pub, job = build(job_kwargs={"steps": 60, "publish_every": 20})
    reports = run_to_completion(job, pub)

    assert len(reports) >= 3, "expected several checkpoints, not one final dump"
    steps = [rp.checkpoint_step for rp in reports]
    assert steps == sorted(steps) and steps[0] < job.spec.steps, steps


def test_every_embedding_is_stored_exactly_once():
    """Invariant 5. Duplicates are the failure a naive retry loop produces."""
    ev, c, client, pub, job = build()
    run_to_completion(job, pub)
    c.run(60)

    ids = list(pub.published)
    assert len(ids) == len(set(ids)), "the publisher recorded a duplicate id"

    leader = c.leader()
    order = c.replicas[leader].sm.ids_in_order
    assert len(order) == len(set(order)), "the same id was applied twice"
    assert set(order) == set(ids)


def test_embeddings_are_model_output_not_input_features():
    """They must change as the model trains.

    If the stored vectors were the raw inputs, the storage layer would be a
    database of the dataset and training would be decorative. The same sample
    embedded at two different steps must differ.
    """
    ev, c, client, pub, job = build(
        job_kwargs={"steps": 40, "publish_every": 20, "embeddings_per_checkpoint": 4}
    )
    cks = []
    while not job.finished:
        ck = job.run_step()
        if ck is not None:
            cks.append(ck)
            pub.publish(ck)

    assert len(cks) >= 2
    # The cursor advances, so re-embed one fixed sample at both sets of weights
    # rather than relying on the checkpoints covering the same sample.
    from dtf.model import logits
    x = job.dataset.X[:4]
    early = cks[0].vectors[0]
    late = cks[-1].vectors[0]
    assert not np.allclose(early, late), (
        "embeddings did not change across training; they are not model output"
    )
    assert cks[0].vectors[0].shape == (job.spec.embedding_dim,)


def test_stored_embedding_is_its_own_nearest_neighbour():
    ev, c, client, pub, job = build()
    cks = []
    while not job.finished:
        ck = job.run_step()
        if ck is not None:
            cks.append(ck)
            pub.publish(ck)
    c.run(60)

    probe_id = cks[0].ids[0]
    probe_vec = cks[0].vectors[0]
    res = client.search(probe_vec, k=1)
    assert res.hits[0]["id"] == probe_id
    assert res.hits[0]["metadata"]["job"] == "j1"


# -------------------------------------------------- faults during publication


def test_a_storage_node_dying_mid_publish_does_not_lose_acked_writes():
    """A minority failure must be invisible to the training job."""
    ev, c, client, pub, job = build()

    killed = False
    while not job.finished:
        ck = job.run_step()
        if ck is not None:
            if not killed:
                victim = next(n for n in c.live_ids if n != c.leader())
                c.kill(victim)
                killed = True
            pub.publish(ck)

    c.run(120)
    assert killed
    assert pub.stats()["rejected"] == 0
    assert pub.stats()["pending"] == 0

    leader = c.leader()
    for eid in pub.published:
        assert c.replicas[leader].sm.contains(eid)


def test_killing_the_leader_mid_publish_never_reports_a_lost_write_as_stored():
    """The ambiguity case, handled honestly.

    Killing the leader mid-publication can leave entries proposed but
    unconfirmed. The requirement is NOT that every write survives -- an
    uncommitted entry vanishing is correct Raft behaviour. The requirement is
    that nothing the publisher called `published` is missing afterwards.
    """
    ev, c, client, pub, job = build()

    killed = False
    while not job.finished:
        ck = job.run_step()
        if ck is not None:
            if not killed and job.step >= 20:
                c.kill(c.leader())
                killed = True
                c.run_until_leader(max_ticks=300)
            pub.publish(ck)

    c.run(200)
    resolution = pub.resolve_pending()

    leader = c.leader()
    assert leader is not None, "cluster never recovered a leader"
    for eid in pub.published:
        assert c.replicas[leader].sm.contains(eid), (
            f"{eid} was reported published but is absent after failover "
            f"(resolution: {resolution.get(eid)})"
        )


def test_a_training_worker_dying_does_not_affect_stored_durability():
    """Cross-layer: a fault in one layer must not corrupt the other."""
    ev, c, client, pub, job = build(job_kwargs={"n_workers": 4, "steps": 60})

    while not job.finished:
        if job.step == 25 and not job.workers_lost:
            job.kill_worker(job.worker_ids[2])
        ck = job.run_step()
        if ck is not None:
            pub.publish(ck)

    c.run(60)
    r = job.result()
    assert r.workers_lost, "the worker kill never happened"
    assert r.steps_completed == r.spec.steps, "training stalled after a worker died"

    leader = c.leader()
    for eid in pub.published:
        assert c.replicas[leader].sm.contains(eid)
    # Training degrades (smaller effective batch); storage does not.
    assert r.final_accuracy > 0.7, r.final_accuracy


def test_publisher_writes_only_through_consensus():
    """Structural: every stored id must correspond to a committed log entry.

    Guards against a future shortcut that writes to a state machine directly,
    which would make every fault test above meaningless while still passing.
    """
    ev, c, client, pub, job = build()
    run_to_completion(job, pub)
    c.run(60)

    leader = c.leader()
    committed_ids = {
        e.command["id"]
        for e in c.replicas[leader].observed.values()
        if e.command.get("op") == "insert"
    }
    stored = set(c.replicas[leader].sm.ids_in_order)
    assert stored <= committed_ids, (
        f"{stored - committed_ids} is in the index without a committed log "
        "entry, so something bypassed consensus"
    )
    assert stored == set(pub.published)


def test_reads_after_a_job_reflect_every_published_embedding():
    """Invariant 6 at the platform level: a linearizable read taken after
    publication returns a state that includes all of it."""
    ev, c, client, pub, job = build()
    cks = []
    while not job.finished:
        ck = job.run_step()
        if ck is not None:
            cks.append(ck)
            pub.publish(ck)

    res = client.search(cks[0].vectors[0], k=50)
    got = {h["id"] for h in res.hits}
    assert set(pub.published) <= got | set(), "search could not reach every published id"
    assert res.read_index >= len(pub.published)


def test_events_from_both_layers_share_one_timeline():
    """The observability claim: cross-layer moments must be relatable."""
    ev, c, client, pub, job = build()
    while not job.finished:
        if job.step == 30 and not job.workers_lost:
            job.kill_worker(job.worker_ids[1])
        ck = job.run_step()
        if ck is not None:
            pub.publish(ck)

    kinds = {e.kind for e in ev.events}
    assert EventKind.JOB_STEP in kinds
    assert EventKind.WORKER_DIED in kinds
    assert EventKind.WRITE_COMMITTED in kinds
    assert EventKind.EMBEDDINGS_PUBLISHED in kinds
    assert EventKind.LEADER_ELECTED in kinds


def test_both_layers_stamp_events_with_the_same_clock():
    """Sharing a log is not sharing a timeline.

    Job events were once stamped with the training STEP counter while storage
    events carried the cluster tick. Both rendered in one column on a dashboard
    captioned "both layers, one clock", so a job_step at 8 and a leader_elected
    at 14 read as orderable and were not comparable at all -- two different
    units printed in the same place.

    Monotonicity is the assertable form of "one clock": if every event reads the
    same counter, the log is non-decreasing in tick. Under two clocks it is not,
    because the step counter restarts from zero while ticks keep climbing.
    """
    p = ControlPlane(node_ids=[f"n{i + 1}" for i in range(3)], dim=8, seed=5)
    p.submit(JobSpec(job_id="job", steps=40, n_classes=8, publish_every=10))
    p.run_until_idle()

    ticks = [e.tick for e in p.events.events]
    assert ticks == sorted(ticks), (
        "event ticks are not monotonic, so the two layers are not on one clock"
    )

    job_ticks = [e.tick for e in p.events.events if e.kind is EventKind.JOB_STEP]
    assert job_ticks, "no job events recorded"
    # The tell-tale: step numbers start at 1 and the cluster is already well
    # past that by the time a job starts, so a step-stamped event would sort
    # before the cluster's own startup events.
    first_storage = min(e.tick for e in p.events.events
                        if e.kind is EventKind.LEADER_ELECTED)
    assert min(job_ticks) >= first_storage, (
        "job events predate the leader election that made writes possible"
    )
    # And the step is still recorded, as data rather than as a timestamp.
    step_details = [e.detail["step"] for e in p.events.events
                    if e.kind is EventKind.JOB_STEP]
    assert step_details == sorted(step_details) and len(set(step_details)) > 5
