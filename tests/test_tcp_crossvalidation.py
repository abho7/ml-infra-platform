"""
The same storage layer, over real sockets.

Everything else in this project runs the cluster on `mlplat.storage.transport`,
a deterministic queue. That is a deliberate choice -- it is what makes a fault
reproducible from a seed -- but it is also a modelling assumption, and an
assumption that never gets checked is just a hope. If `VectorStateMachine` only
behaves because messages arrive in a tidy order that a real network would never
produce, every result on the report is about the harness rather than the system.

So these tests run three `kvstore.RaftServer` instances as real asyncio TCP
servers on localhost, with the platform's own state machine attached, and drive
them through the engine's own `KVClient`. Real sockets, real concurrency, real
`asyncio.sleep` timing, no shared memory between the replication paths.

HOW THE STATE MACHINE GETS IN. `RaftServer.__init__` hardcodes
`self.state_machine = KVStateMachine()`, and the engine is read-only, so it is
replaced on the instance after construction. That is driving the engine from
outside, which is the same discipline as everywhere else here -- the engine
file is untouched, and the swap works precisely because `VectorStateMachine`
implements the interface `RaftServer` actually calls: `apply_all(entries)` and
`last_applied_index`.

These are marked slow. They involve real election timeouts and real sleeps, so
they take seconds rather than milliseconds, and they are not deterministic in
the way the rest of the suite is.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket

import numpy as np
import pytest

import mlplat.enginepath  # noqa: F401
from kvstore.client import KVClient
from kvstore.server import RaftServer

from mlplat.storage.statemachine import VectorStateMachine

DIM = 8
pytestmark = pytest.mark.slow


def _free_ports(n: int) -> list[int]:
    """Bind to port 0 and let the OS choose, rather than guessing.

    Hardcoded ports make a test that fails on whatever else happens to be
    listening, which is a flake that looks like a bug in the system.
    """
    socks = []
    try:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            socks.append(s)
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


async def _cluster(n: int = 3):
    ports = _free_ports(n)
    ids = [f"t{i + 1}" for i in range(n)]
    addrs = {nid: ("127.0.0.1", p) for nid, p in zip(ids, ports)}

    servers = {}
    for nid in ids:
        host, port = addrs[nid]
        srv = RaftServer(
            nid, {k: v for k, v in addrs.items() if k != nid}, host, port,
            election_timeout_ticks=(6, 12),
            heartbeat_interval_ticks=2,
            tick_interval_seconds=0.02,
        )
        # The swap. Same seed on every replica, exactly as StorageCluster does,
        # so the graphs are built identically from an identical log.
        srv.state_machine = VectorStateMachine(dim=DIM, seed=20260825)
        await srv.start()
        servers[nid] = srv
    return servers, addrs


async def _await_leader(servers, timeout=8.0):
    from raft.node import Role
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        leaders = [n for n, s in servers.items() if s.node.role is Role.LEADER]
        if len(leaders) == 1:
            return leaders[0]
        await asyncio.sleep(0.05)
    raise AssertionError("no leader elected over real TCP within the timeout")


async def _stop(servers):
    for s in servers.values():
        with contextlib.suppress(Exception):
            await s.stop()


def _vec(i: int) -> list[float]:
    rng = np.random.default_rng(i)
    return rng.normal(size=DIM).tolist()


# ==========================================================================


def test_embeddings_replicate_identically_over_real_tcp():
    """The core cross-validation.

    If every replica ends up with the same HNSW graph after writes that crossed
    real sockets in real time, then the deterministic transport used everywhere
    else is a fair model of this for the property being claimed.
    """
    async def main():
        servers, addrs = await _cluster(3)
        try:
            await _await_leader(servers)
            client = KVClient(addrs, request_timeout_seconds=3.0, max_retries=10)

            written = []
            for i in range(12):
                eid = f"tcp-{i}"
                await client._request(
                    {"op": "insert", "id": eid, "vector": _vec(i),
                     "metadata": {"job": "tcp", "i": i}}
                )
                written.append(eid)

            # Let the followers finish applying; commits reach them on the next
            # heartbeat, which is real elapsed time here rather than a tick.
            await asyncio.sleep(1.0)

            digests = {n: s.state_machine.digest() for n, s in servers.items()}
            sizes = {n: s.state_machine.size for n, s in servers.items()}
            return written, digests, sizes
        finally:
            await _stop(servers)

    written, digests, sizes = asyncio.run(main())

    assert len(set(digests.values())) == 1, (
        f"replicas diverged over real TCP: {digests}"
    )
    assert set(sizes.values()) == {len(written)}, sizes


def test_the_duplicate_id_bug_stays_fixed_over_real_tcp():
    """The composition bug, re-checked on the transport it was never seen on.

    A client that re-proposes an unconfirmed write can get the same embedding id
    committed at two log indices. Before the fix that raised inside apply and
    stopped every replica dead. Here the duplicate is proposed deliberately and
    on purpose: the cluster must commit both entries (Raft was asked twice, and
    it does what it is asked) while the state machine stores the embedding once
    and keeps running.
    """
    async def main():
        servers, addrs = await _cluster(3)
        try:
            leader = await _await_leader(servers)
            client = KVClient(addrs, request_timeout_seconds=3.0, max_retries=10)

            cmd = {"op": "insert", "id": "dup", "vector": _vec(1), "metadata": {}}
            await client._request(cmd)
            await client._request(dict(cmd))       # the same id, a second entry
            await client._request(
                {"op": "insert", "id": "after", "vector": _vec(2), "metadata": {}}
            )
            await asyncio.sleep(1.0)

            sm = servers[leader].state_machine
            return {
                "log_len": servers[leader].node.log.last_index,
                "size": sm.size,
                "dupes": sm.duplicates_suppressed,
                "has_after": sm.contains("after"),
                "digests": {n: s.state_machine.digest() for n, s in servers.items()},
            }
        finally:
            await _stop(servers)

    r = asyncio.run(main())

    assert r["size"] == 2, f"expected 2 embeddings stored, got {r['size']}"
    assert r["dupes"] == 1, "the duplicate was not suppressed"
    assert r["has_after"], (
        "the entry after the duplicate never applied -- apply died on the "
        "duplicate, which is the original bug"
    )
    assert len(set(r["digests"].values())) == 1, r["digests"]


def test_a_write_survives_losing_a_node_over_real_tcp():
    """Minority failure, on a real network.

    The surviving replicas must agree, and everything acknowledged before the
    failure must still be there afterwards.
    """
    async def main():
        servers, addrs = await _cluster(3)
        try:
            leader = await _await_leader(servers)
            client = KVClient(addrs, request_timeout_seconds=3.0, max_retries=12)

            before = []
            for i in range(5):
                eid = f"pre-{i}"
                await client._request(
                    {"op": "insert", "id": eid, "vector": _vec(i), "metadata": {}})
                before.append(eid)
            await asyncio.sleep(0.6)

            victim = next(n for n in servers if n != leader)
            await servers[victim].stop()

            after = []
            for i in range(5):
                eid = f"post-{i}"
                await client._request(
                    {"op": "insert", "id": eid, "vector": _vec(50 + i), "metadata": {}})
                after.append(eid)
            await asyncio.sleep(1.0)

            alive = {n: s for n, s in servers.items() if n != victim}
            return {
                "before": before, "after": after,
                "digests": {n: s.state_machine.digest() for n, s in alive.items()},
                "present": {
                    n: all(s.state_machine.contains(e) for e in before + after)
                    for n, s in alive.items()
                },
            }
        finally:
            await _stop(servers)

    r = asyncio.run(main())

    assert all(r["present"].values()), (
        f"an acknowledged write is missing after a node died: {r['present']}"
    )
    assert len(set(r["digests"].values())) == 1, (
        f"survivors diverged after the failure: {r['digests']}"
    )
