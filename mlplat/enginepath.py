"""
Puts the three read-only engines on sys.path.

Imported for its side effect by every module that touches an engine. Doing it
in one place rather than per-module means there is exactly one description of
where the engines live, and exactly one place to look when an import fails.

The three repos use different layouts, which is why this is not a one-liner:

    raft-engine/src        ->  raft.*, kvstore.*      (src layout)
    hnsw-engine/src        ->  hnsw.*, vectordb.*     (src layout)
    training-engine/       ->  dtf.*, engines.*, chaos.*   (flat layout)

NAMESPACE HAZARD. The training engine occupies the top-level names `dtf`,
`engines` and `chaos`. This platform's own code therefore lives entirely under
`mlplat.*` -- in particular `mlplat.chaos` is a different module from the
training engine's `chaos`, and a top-level `chaos/` directory here would
shadow the engine's and break `chaos.runner` imports in a way that looks like a
missing file rather than a collision.

The engines are dependencies, not vendored source: nothing here modifies them,
and they are cloned rather than copied so the platform always builds against
their current main.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAFT_ENGINE = ROOT / "raft-engine"
HNSW_ENGINE = ROOT / "hnsw-engine"
TRAINING_ENGINE = ROOT / "training-engine"

_PATHS = [
    RAFT_ENGINE / "src",
    HNSW_ENGINE / "src",
    TRAINING_ENGINE,
]


def install() -> None:
    """Idempotent. Safe to call from any module, in any order."""
    missing = [p for p in _PATHS if not p.exists()]
    if missing:
        raise RuntimeError(
            "read-only engine dependencies are not present: "
            + ", ".join(str(p) for p in missing)
            + "\n\nClone them first:\n"
            "  git clone https://github.com/abho7/raft-kv-store.git ./raft-engine\n"
            "  git clone https://github.com/abho7/vectordb-hnsw.git ./hnsw-engine\n"
            "  git clone https://github.com/abho7/distributed-training-framework.git"
            " ./training-engine"
        )
    for p in _PATHS:
        s = str(p)
        if s not in sys.path:
            # Appended, not prepended: this platform's own packages must win
            # any name contest with an engine, not the other way round.
            sys.path.append(s)


install()
