"""
Records a live session so it can be replayed later.

The deployed dashboard is a static page and cannot run a Raft cluster, so it
replays a recording instead. This is what makes that honest: a recording is
captured from an actual run of the real platform -- the same ControlPlane, the
same faults, the same checker -- and every frame is a snapshot that existed.
Nothing on the replay is generated for the replay.

The distinction matters and is stated on the page itself. A "live demo" that is
secretly a scripted animation is a lie; a recording labelled as a recording is
just a recording.

FRAMES ARE SNAPSHOTS, NOT DELTAS. Storing full state per frame is larger, but a
scrubber that can jump to any point needs a complete state at that point, and
reconstructing one by replaying deltas from the start is exactly the kind of
thing that quietly desynchronises. At these sizes the file is small enough that
the simpler option wins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Recording:
    title: str
    description: str
    frames: list[dict] = field(default_factory=list)
    faults: list[dict] = field(default_factory=list)
    result: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "frames": self.frames,
            "faults": self.faults,
            "result": self.result,
        }


class SessionRecorder:
    """Captures a frame per round from a running ControlPlane."""

    def __init__(self, plane, *, title: str, description: str) -> None:
        self.plane = plane
        self.recording = Recording(title=title, description=description)
        self._last_event = 0

    def capture(self, rnd: int, *, note: str | None = None) -> None:
        s = self.plane.state()
        # Only the events since the previous frame. The full stream is on the
        # results payload; a frame carries what happened *at* it, so the replay
        # can show the moment a node died rather than re-listing history.
        new_events = [
            e.to_json() for e in self.plane.events.events[self._last_event:]
        ]
        self._last_event = len(self.plane.events.events)

        self.recording.frames.append({
            "round": rnd,
            "tick": s["tick"],
            "health": s["health"],
            "nodes": s["cluster"]["nodes"],
            "partition": s["cluster"]["partition"],
            "jobs": [
                {
                    "job_id": j["job_id"], "state": j["state"],
                    "step": j["step"], "total_steps": j["total_steps"],
                    "loss": j["loss"], "published": j["published"],
                    "pending": j["pending"], "workers": j["workers"],
                    "workers_lost": j["workers_lost"],
                }
                for j in s["jobs"]
            ],
            "storage": s["storage"],
            "events": new_events,
            "note": note,
        })

    def finish(self, result: dict, faults: list[dict]) -> Recording:
        self.recording.result = result
        self.recording.faults = faults
        return self.recording

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.recording.to_json(), default=str),
                        encoding="utf-8")
        return path
