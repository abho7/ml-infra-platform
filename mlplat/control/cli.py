"""
`mlplat` -- the command line control plane.

Two modes, and the distinction matters:

  SERVER MODE   `mlplat serve` runs the HTTP control plane and the dashboard.
                Every other command then talks to it over HTTP, so the CLI and
                the dashboard are looking at the same live system.

  LOCAL MODE    `mlplat demo` builds a plane in-process and drives it to
                completion. For CI and for reproducing a scenario exactly,
                where a background ticker would make runs non-deterministic.

Commands that mutate the cluster (kill, partition) are first-class rather than
hidden behind a test-only flag. Fault injection is a feature of this platform,
not scaffolding for its tests -- the whole claim is about behaviour under
failure, so the control plane has to be able to cause failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8950"


def _get(url: str, path: str) -> dict:
    with urllib.request.urlopen(url + path, timeout=10) as r:
        return json.loads(r.read())


def _post(url: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


# ------------------------------------------------------------------ rendering

BAR = "#"


def _health_line(h: dict) -> str:
    mark = {"healthy": "OK", "degraded": "!!", "unavailable": "XX"}[h["status"]]
    return (
        f"[{mark}] {h['status']:<12} leader={h['leader'] or '(none)':<6} "
        f"live={h['live_nodes']}/{h['total_nodes']} "
        f"quorum={'yes' if h['has_quorum'] else 'NO'} "
        f"(need {h['quorum_needed']}, largest group {h['largest_reachable_group']})"
        + ("  PARTITIONED" if h["partitioned"] else "")
    )


def _render_state(s: dict) -> str:
    out = [_health_line(s["health"]), ""]
    out.append(f"{'node':<6}{'role':<11}{'term':>5}{'log':>6}{'commit':>8}"
               f"{'applied':>9}{'vectors':>9}")
    for nid, n in s["cluster"]["nodes"].items():
        role = n["role"] if n["alive"] else "DEAD"
        out.append(f"{nid:<6}{role:<11}{n['term']:>5}{n['log_len']:>6}"
                   f"{n['commit_index']:>8}{n['applied_index']:>9}{n['vectors']:>9}")

    if s["jobs"]:
        out += ["", f"{'job':<10}{'state':<12}{'progress':<22}{'loss':>9}"
                    f"{'published':>11}{'pending':>9}"]
        for j in s["jobs"]:
            pct = j["progress"]
            bar = BAR * int(pct * 16)
            prog = f"[{bar:<16}] {j['step']}/{j['total_steps']}"
            loss = f"{j['loss']:.4f}" if j["loss"] is not None else "-"
            out.append(f"{j['job_id']:<10}{j['state']:<12}{prog:<22}{loss:>9}"
                       f"{j['published']:>11}{j['pending']:>9}")

    st = s["storage"]
    out += ["", f"tick={s['tick']}  acked_writes={st['acked_writes']}  "
                f"reads issued={st['reads']['issued']} refused={st['reads']['refused']}  "
                f"events={s['events']}"]
    return "\n".join(out)


# ------------------------------------------------------------------ commands


def cmd_serve(args) -> int:
    from mlplat.control.api import serve
    from mlplat.control.plane import ControlPlane

    plane = ControlPlane(
        node_ids=[f"n{i + 1}" for i in range(args.nodes)], dim=args.dim, seed=args.seed
    )
    httpd, ticker = serve(plane, host=args.host, port=args.port, hz=args.hz)
    print(f"mlplat control plane on http://{args.host}:{args.port}")
    print(f"  dashboard   http://{args.host}:{args.port}/")
    print(f"  cluster     {args.nodes} nodes, dim={args.dim}, leader={plane.cluster.leader()}")
    print(f"  ticking at  {args.hz} Hz   (ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        ticker.stop()
        httpd.shutdown()
    return 0


def cmd_status(args) -> int:
    print(_render_state(_get(args.url, "/api/state")))
    return 0


def cmd_watch(args) -> int:
    """Poll the live plane and redraw. The terminal counterpart of the
    dashboard, for when a browser is not where you are working."""
    try:
        while True:
            s = _get(args.url, "/api/state")
            sys.stdout.write("\x1b[2J\x1b[H" + _render_state(s) + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def cmd_submit(args) -> int:
    spec = {
        "job_id": args.job_id, "n_workers": args.workers, "steps": args.steps,
        "n_classes": args.dim, "strategy": args.strategy,
        "publish_every": args.publish_every,
    }
    print(json.dumps(_post(args.url, "/api/jobs", spec), indent=1))
    return 0


def cmd_kill(args) -> int:
    print(json.dumps(_post(args.url, "/api/nodes/kill", {"node": args.node}), indent=1))
    return 0


def cmd_restart(args) -> int:
    print(json.dumps(_post(args.url, "/api/nodes/restart", {"node": args.node}), indent=1))
    return 0


def cmd_partition(args) -> int:
    groups = [g.split(",") for g in args.groups]
    print(json.dumps(_post(args.url, "/api/partition", {"groups": groups}), indent=1))
    return 0


def cmd_heal(args) -> int:
    print(json.dumps(_post(args.url, "/api/heal", {}), indent=1))
    return 0


def cmd_events(args) -> int:
    for e in _get(args.url, f"/api/events?limit={args.limit}")["events"]:
        detail = " ".join(f"{k}={v}" for k, v in e["detail"].items())
        print(f"  t={e['tick']:<6}{e['kind']:<24}{detail}")
    return 0


def cmd_demo(args) -> int:
    """In-process, deterministic, no server. What CI runs."""
    from mlplat.control.plane import ControlPlane
    from mlplat.training.job import JobSpec

    plane = ControlPlane(
        node_ids=[f"n{i + 1}" for i in range(args.nodes)], dim=args.dim, seed=args.seed
    )
    plane.submit(JobSpec(job_id="demo", n_workers=args.workers,
                         steps=args.steps, n_classes=args.dim))
    print(_health_line(plane.state()["health"]))
    rounds = plane.run_until_idle()
    print(f"\ncompleted in {rounds} rounds\n")
    print(_render_state(plane.state()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlplat", description=__doc__.split("\n")[1])
    p.add_argument("--url", default=DEFAULT_URL, help="control plane address")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the control plane and dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8950)
    s.add_argument("--nodes", type=int, default=5)
    s.add_argument("--dim", type=int, default=8)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--hz", type=float, default=10.0)
    s.set_defaults(fn=cmd_serve)

    for name, fn, helptext in (
        ("status", cmd_status, "print cluster and job state once"),
        ("heal", cmd_heal, "heal any active partition"),
    ):
        c = sub.add_parser(name, help=helptext)
        c.set_defaults(fn=fn)

    w = sub.add_parser("watch", help="live-redraw the state in the terminal")
    w.add_argument("--interval", type=float, default=0.5)
    w.set_defaults(fn=cmd_watch)

    j = sub.add_parser("submit", help="submit a training job")
    j.add_argument("job_id")
    j.add_argument("--workers", type=int, default=4)
    j.add_argument("--steps", type=int, default=60)
    j.add_argument("--dim", type=int, default=8)
    j.add_argument("--strategy", default="all-reduce",
                   choices=["all-reduce", "parameter-server"])
    j.add_argument("--publish-every", type=int, default=20)
    j.set_defaults(fn=cmd_submit)

    for name, fn in (("kill", cmd_kill), ("restart", cmd_restart)):
        c = sub.add_parser(name, help=f"{name} a storage node")
        c.add_argument("node")
        c.set_defaults(fn=fn)

    pt = sub.add_parser("partition", help="partition the cluster, e.g. n1,n2 n3,n4,n5")
    pt.add_argument("groups", nargs="+")
    pt.set_defaults(fn=cmd_partition)

    e = sub.add_parser("events", help="recent events from every layer")
    e.add_argument("--limit", type=int, default=40)
    e.set_defaults(fn=cmd_events)

    d = sub.add_parser("demo", help="run a job to completion in-process")
    d.add_argument("--nodes", type=int, default=5)
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--steps", type=int, default=60)
    d.add_argument("--dim", type=int, default=8)
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(fn=cmd_demo)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except urllib.error.URLError:
        print(f"cannot reach a control plane at {args.url}\n"
              f"start one with:  mlplat serve", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
