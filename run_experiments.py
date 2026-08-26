"""
Runs everything that produces a number on the report, and writes the results.

This is the only source of `site/results.json`. Nothing on the deployed page is
typed in by hand; if a figure is on the site, this script measured it, and
re-running this script is how you check that.

It also records one real session (`site/session.json`) for the dashboard to
replay: the same ControlPlane, driven through a scripted set of faults, with a
frame captured every round.

    python run_experiments.py            # everything
    python run_experiments.py --quick    # smaller sweep, for iterating
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import mlplat.enginepath  # noqa: F401
from mlplat.chaos import scenarios as scen
from mlplat.chaos.fuzz import sweep
from mlplat.control.plane import ControlPlane, JobState
from mlplat.observe.events import EventLog
from mlplat.observe.recorder import SessionRecorder
from mlplat.training.job import JobSpec

sys.path.insert(0, str(Path(__file__).parent / "bench"))
import run_bench  # noqa: E402

SITE = Path(__file__).parent / "site"


# ------------------------------------------------------------------ session

def record_session(*, dim: int = 8) -> dict:
    """One real run, driven through faults, captured frame by frame.

    The faults are scripted rather than random for the obvious reason: the
    replay is meant to show a viewer something legible, and a randomized run
    mostly shows a cluster being fine. Scripting *when* the faults happen does
    not script what the system does about them, which is the part being shown.
    """
    events = EventLog()
    plane = ControlPlane(node_ids=[f"n{i + 1}" for i in range(5)],
                         dim=dim, seed=7, events=events)
    # (round, kind, description, action). `kind` separates a fault from a
    # recovery so the title can COUNT the faults rather than assert a number
    # that drifts the moment the script changes -- the page prints this title
    # verbatim, so a hardcoded count is a claim waiting to go stale.
    script = [
        (14, "fault", "kill a follower",
         lambda: plane.kill_node(_a_follower(plane))),
        (30, "fault", "partition the cluster 2 | 3",
         lambda: _split(plane)),
        (36, "fault", "kill a training worker, mid-partition",
         lambda: plane.kill_worker("job")),
        (58, "recovery", "heal the partition",
         plane.heal),
        (70, "recovery", "restart the dead node",
         lambda: [plane.restart_node(n) for n in list(plane.cluster.dead_ids)]),
    ]
    n_faults = sum(1 for e in script if e[1] == "fault")

    rec = SessionRecorder(
        plane,
        title=(f"five nodes, one training job, {n_faults} injected faults "
               f"across both layers"),
        description=(
            "A recorded session: a 4-worker training job publishing embeddings "
            "through Raft into the HNSW store, while a node is killed, the "
            "cluster is partitioned, and a training worker dies during the "
            "partition. Every frame is a snapshot of a real run."
        ),
    )
    plane.submit(JobSpec(job_id="job", n_workers=4, steps=90,
                         n_classes=dim, publish_every=10, seed=107))

    faults = []
    pending = {r: (k, d, f) for r, k, d, f in script}

    rnd = 0
    while any(m.state not in (JobState.DONE, JobState.FAILED)
              for m in plane.jobs.values()):
        note = None
        if rnd in pending:
            kind, desc, fn = pending.pop(rnd)
            try:
                fn()
                note = desc
                faults.append({"round": rnd, "tick": plane.cluster.tick,
                               "kind": kind, "description": desc})
            except Exception as exc:           # a fault that cannot fire is data
                note = f"{desc} (skipped: {exc})"
        plane.advance()
        rec.capture(rnd, note=note)
        rnd += 1
        if rnd > 1200:
            break

    # Settle, exactly as the chaos runner does, and capture the recovery.
    plane.heal()
    for nid in list(plane.cluster.dead_ids):
        plane.restart_node(nid)
    for i in range(40):
        plane.cluster.run(8)
        rec.capture(rnd + i, note="settling" if i == 0 else None)

    managed = plane.job("job")
    managed.publisher.resolve_pending()
    leader = plane.cluster.leader()
    published = set(managed.publisher.published)
    queryable = {e for e in published
                 if plane.cluster.replicas[leader].sm.contains(e)} if leader else set()
    digests = plane.cluster.digests()

    result = {
        "rounds": rnd,
        "published": len(published),
        "queryable": len(queryable),
        "lost_uncommitted": len(managed.publisher.rejected),
        "converged": len(set(digests.values())) <= 1,
        "final_leader": leader,
        "accuracy": managed.job.result().final_accuracy,
        "events": len(events),
    }
    rec.finish(result, faults)
    rec.write(SITE / "session.json")
    print(f"  session: {len(rec.recording.frames)} frames, "
          f"{result['published']} published, {result['queryable']} queryable, "
          f"converged={result['converged']}")
    return {**result, "frames": len(rec.recording.frames),
            "faults": faults, "title": rec.recording.title}


def _a_follower(plane) -> str:
    leader = plane.cluster.leader()
    return next(n for n in plane.cluster.live_ids if n != leader)


def _split(plane) -> None:
    ids = list(plane.cluster.node_ids)
    plane.partition(set(ids[:2]), set(ids[2:]))


# ------------------------------------------------------------------- driver

def run_suite() -> dict:
    """Run pytest and record what it actually reported.

    The report says how many tests pass. That sentence is only worth printing
    if it comes from a run rather than from memory, so the number on the page
    traces to this subprocess like every other figure.
    """
    import re
    import subprocess

    # `-o addopts=` CLEARS what pytest.ini already sets. Without it the ini's
    # own `-q` combines with a second one into `-qq`, which suppresses the
    # summary line entirely -- the parse below then found nothing and this
    # returned passed=0 while exit_code stayed 0. The page printed "0 tests
    # passing" as though that had been measured, which is precisely the silent
    # wrong number the rest of this file exists to prevent.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "--tb=no", "-q"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent),
    )
    out = proc.stdout or ""
    got = {k: int(v) for v, k in
           re.findall(r"(\d+) (passed|failed|error|errors|skipped)", out)}
    passed = got.get("passed", 0)
    failed = got.get("failed", 0) + got.get("error", 0) + got.get("errors", 0)

    if passed == 0:
        # Nothing parsed. Refuse to report a number rather than report zero.
        raise RuntimeError(
            "could not read a test count out of pytest's output; refusing to "
            f"publish a fabricated one.\nexit={proc.returncode}\n"
            f"last lines:\n" + "\n".join(out.strip().splitlines()[-5:])
        )

    summary = next((ln for ln in reversed(out.strip().splitlines())
                    if "passed" in ln or "failed" in ln), "")
    return {
        "passed": passed,
        "failed": failed,
        "skipped": got.get("skipped", 0),
        "exit_code": proc.returncode,
        "summary": summary.strip(),
    }


def main() -> None:
    quick = "--quick" in sys.argv
    SITE.mkdir(exist_ok=True)
    t0 = time.time()

    print("[1/6] recorded session")
    session = record_session()

    print("[2/6] named cross-layer scenarios")
    results = [s.to_json() for s in scen.run_all()]
    for r in results:
        print(f"  {r['slug']:<34} {'PASS' if r['passed'] else 'FAIL'}  "
              f"published={r['published']} queryable={r['queryable']} "
              f"converged={r['converged']}")

    print("[3/6] randomized cross-layer sweep")
    fuzz = sweep(12 if quick else 60, quiet=True)
    print(f"  {fuzz['passed']}/{fuzz['runs']} seeds clean, "
          f"{fuzz['total_published']} embeddings published, "
          f"{fuzz['published_not_queryable']} not queryable")

    print("[4/6] benchmarks")
    bench = run_bench.run_all()

    print("[5/6] full test suite")
    suite = run_suite()
    print(f"  {suite['summary'] or 'no summary line'}")

    print("[6/6] writing site/results.json")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 1),
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "session": session,
        "scenarios": results,
        "fuzz": fuzz,
        "bench": bench,
        "suite": suite,
    }
    (SITE / "results.json").write_text(
        json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"done in {payload['elapsed_s']}s")


if __name__ == "__main__":
    main()
