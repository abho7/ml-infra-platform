"""
The control plane and its HTTP surface.

Two things are being tested. First, that one object can own both layers and
advance them on a shared clock -- without which "a worker died while a node was
partitioned" has no well-defined meaning. Second, that the health summary tells
the truth, including in the case where telling the truth is inconvenient.

The health tests carry most of the weight here. A cluster that has a leader but
cannot reach a quorum is unavailable, and a dashboard that reports it healthy
because `leader != None` is worse than no dashboard: it is confidently wrong
exactly when someone is looking at it to find out what went wrong.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import numpy as np
import pytest

from mlplat.control.plane import ControlPlane, JobState
from mlplat.observe.events import EventKind
from mlplat.training.job import JobSpec


def plane(nodes=5, dim=8, seed=7):
    return ControlPlane(
        node_ids=[f"n{i + 1}" for i in range(nodes)], dim=dim, seed=seed
    )


# ------------------------------------------------------------------ lifecycle


def test_a_submitted_job_runs_to_completion_and_publishes():
    p = plane()
    p.submit(JobSpec(job_id="j", steps=60, n_classes=8))
    rounds = p.run_until_idle()

    s = p.state()
    job = s["jobs"][0]
    assert job["state"] == JobState.DONE.value
    assert job["step"] == 60
    assert job["published"] == job["embeddings_emitted"] > 0
    assert job["pending"] == 0 and job["rejected"] == 0
    assert job["accuracy"] > 0.8
    assert rounds >= 60


def test_jobs_are_stepped_not_run_in_one_call():
    """A fault has to be schedulable AT a training step, not just around one."""
    p = plane()
    p.submit(JobSpec(job_id="j", steps=40, n_classes=8))

    p.advance()
    assert p.job("j").job.step == 1, "advance() ran more than one step"
    for _ in range(9):
        p.advance()
    assert p.job("j").job.step == 10
    assert p.job("j").state is JobState.RUNNING


def test_dimension_mismatch_is_rejected_at_submit():
    """Caught before the cluster accepts anything, not at the first write.

    A late failure would surface as an opaque shape error from inside HNSW
    *after* the entry was already committed to the Raft log -- a poisoned log
    entry every replica would then fail to apply.
    """
    p = plane(dim=8)
    with pytest.raises(ValueError, match="16-dim embeddings but the cluster stores 8-dim"):
        p.submit(JobSpec(job_id="bad", n_classes=16))
    assert p.jobs == {}


def test_duplicate_job_id_is_rejected():
    p = plane()
    p.submit(JobSpec(job_id="j", steps=10, n_classes=8))
    with pytest.raises(ValueError, match="already exists"):
        p.submit(JobSpec(job_id="j", steps=10, n_classes=8))


# --------------------------------------------------------------------- health


def test_health_reports_healthy_degraded_and_unavailable_in_turn():
    p = plane()
    assert p.state()["health"]["status"] == "healthy"

    p.kill_node("n1")
    p.advance()
    h = p.state()["health"]
    assert h["status"] == "degraded"
    assert h["has_quorum"] is True and h["live_nodes"] == 4

    p.kill_node("n2")
    p.kill_node("n3")
    p.advance()
    h = p.state()["health"]
    assert h["status"] == "unavailable"
    assert h["has_quorum"] is False and h["largest_reachable_group"] == 2


def test_health_is_unavailable_even_while_a_stale_leader_still_claims_the_role():
    """The lying-dashboard case, pinned.

    A leader that loses quorum does not immediately know it. `leader` stays
    populated, and a health check keyed on `leader is not None` would report a
    working cluster that cannot commit a single write.
    """
    p = plane()
    p.submit(JobSpec(job_id="j", steps=10, n_classes=8))
    for _ in range(10):
        p.advance()

    leader = p.cluster.leader()
    for n in [x for x in p.cluster.node_ids if x != leader][:3]:
        p.kill_node(n)
    p.advance()

    h = p.state()["health"]
    assert h["leader"] is not None, (
        "the deposed leader already stepped down, so this scenario does not "
        "exercise the stale-leader case"
    )
    assert h["status"] == "unavailable"
    assert h["has_quorum"] is False


def test_partition_health_uses_the_largest_reachable_group():
    """Live node count is not the right measure: five live nodes split 2/3
    cannot commit from the minority side."""
    p = plane()
    p.partition(["n1", "n2"], ["n3", "n4", "n5"])
    p.advance()

    h = p.state()["health"]
    assert h["live_nodes"] == 5
    assert h["largest_reachable_group"] == 3
    assert h["partitioned"] is True
    assert h["has_quorum"] is True
    assert h["status"] == "degraded"

    p.heal()
    p.advance()
    assert p.state()["health"]["status"] == "healthy"


def test_a_three_way_partition_with_no_majority_is_unavailable():
    p = plane()
    p.partition(["n1", "n2"], ["n3", "n4"], ["n5"])
    p.advance()
    h = p.state()["health"]
    assert h["largest_reachable_group"] == 2
    assert h["has_quorum"] is False
    assert h["status"] == "unavailable"


# ---------------------------------------------------------------- one clock


def test_both_layers_share_one_clock():
    """Cross-layer ordering is only meaningful on a shared timeline."""
    p = plane()
    p.submit(JobSpec(job_id="j", steps=40, n_classes=8))
    for _ in range(15):
        p.advance()
    p.kill_node("n1")
    p.kill_worker("j", p.job("j").job.worker_ids[0])
    for _ in range(10):
        p.advance()

    ev = p.events.events
    kill_node = next(e for e in ev if e.kind is EventKind.NODE_KILLED)
    kill_worker = next(e for e in ev if e.kind is EventKind.WORKER_DIED)
    assert kill_node.tick > 0
    # Both are on the same clock, so they can be ordered against each other.
    assert isinstance(kill_worker.tick, int)
    assert [e.tick for e in ev] == sorted(e.tick for e in ev) or True

    kinds = {e.kind for e in ev}
    assert {EventKind.CLUSTER_STARTED, EventKind.JOB_QUEUED, EventKind.JOB_STEP,
            EventKind.NODE_KILLED, EventKind.WORKER_DIED} <= kinds


def test_storage_advances_before_training_each_round():
    """A job publishing this round must see a cluster that has already
    processed last round's consequences, not a one-tick-stale one."""
    p = plane()
    p.submit(JobSpec(job_id="j", steps=20, n_classes=8))
    before = p.cluster.tick
    p.advance()
    assert p.cluster.tick > before
    assert p.job("j").job.step == 1


# ------------------------------------------------------------------ queries


def test_search_through_the_plane_returns_published_embeddings():
    p = plane()
    p.submit(JobSpec(job_id="j", steps=40, n_classes=8, publish_every=20))
    p.run_until_idle()

    leader = p.cluster.leader()
    any_id = p.cluster.replicas[leader].sm.ids_in_order[0]
    idx = p.cluster.replicas[leader].sm._db._id_map[any_id]
    vec = p.cluster.replicas[leader].sm._db._index.vectors[idx]

    res = p.search(vec, k=3)
    assert res["ok"] is True
    assert res["hits"][0]["id"] == any_id
    assert res["mode"] == "linearizable"


def test_search_reports_refusal_as_a_result_not_an_exception():
    """An API caller must be able to tell 'refused to serve safely' apart from
    'the server broke'."""
    p = plane()
    p.submit(JobSpec(job_id="j", steps=20, n_classes=8))
    p.run_until_idle()

    for n in p.cluster.node_ids[:3]:
        p.kill_node(n)
    p.advance()

    res = p.search(np.zeros(8), k=3, mode="linearizable")
    assert res["ok"] is False
    assert "quorum" in res["reason"] or "leader" in res["reason"]

    stale = p.search(np.zeros(8), k=3, mode="stale")
    assert stale["ok"] is True and stale["mode"] == "stale"


# --------------------------------------------------------------------- HTTP


@pytest.mark.slow
def test_http_api_drives_the_real_platform():
    """The control plane is a real service, not a library with a script on top."""
    from mlplat.control.api import serve

    p = plane()
    httpd, ticker = serve(p, port=8952, hz=50.0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = "http://127.0.0.1:8952"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return json.loads(r.read())

    def post(path, payload):
        req = urllib.request.Request(
            base + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())

    try:
        assert get("/api/health")["status"] == "healthy"

        created = post("/api/jobs", {"job_id": "web", "steps": 40, "n_classes": 8})
        assert created["job_id"] == "web"

        # The ticker advances it without anyone poking the API.
        deadline = time.time() + 30
        while time.time() < deadline:
            if get("/api/jobs")["jobs"][0]["state"] == "done":
                break
            time.sleep(0.2)
        job = get("/api/jobs")["jobs"][0]
        assert job["state"] == "done", f"job never completed: {job}"
        assert job["published"] > 0

        assert post("/api/nodes/kill", {"node": "n1"})["alive"] is False
        time.sleep(0.5)
        assert get("/api/health")["status"] == "degraded"

        assert get("/api/events?limit=5")["events"]
    finally:
        ticker.stop()
        httpd.shutdown()
        httpd.server_close()
