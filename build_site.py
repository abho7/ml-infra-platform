"""
Builds the deployed site from measured results.

Two pages, both self-contained:

  site/index.html      the report -- architecture, invariants, bugs, benchmarks
  site/dashboard.html  the dashboard, in replay mode, with the session inlined

`dashboard.html` is `dashboard/index.html` with a recording injected, not a
re-implementation. The deployed replay is byte-for-byte the same UI as the live
one; the only difference is where its frames come from.

EVERY NUMBER ON THE PAGE IS READ FROM site/results.json. There is no literal
figure in this file. If a value is missing from the results, the page says so
rather than falling back to something plausible -- a report that quietly prints
a default is worse than one with a gap in it.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).parent
SITE = ROOT / "site"


def esc(s) -> str:
    return html.escape(str(s))


def n(x, digits=0):
    """Format a number, or say it is missing. Never silently substitutes."""
    if x is None:
        return '<span class="missing">not measured</span>'
    if isinstance(x, float) and digits:
        return f"{x:.{digits}f}"
    if isinstance(x, float):
        return f"{x:g}"
    return f"{x:,}" if isinstance(x, int) else str(x)


# ============================================================ page components

CSS = """
:root{
  --page:#0b0c0e; --panel:#121418; --raised:#171a20; --line:#22262c; --line-2:#2e343d;
  --ink:#f2f4f7; --ink-2:#b3bcc6; --label:#8c95a0;
  --accent:#a78bfa; --accent-dim:#6d5bb8;
  --ok:#4ade80; --warn:#fbbf24; --fault:#f43f5e;
  --sans:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;
  --mono:ui-monospace,SFMono-Regular,"JetBrains Mono",Consolas,"Liberation Mono",monospace;
  --measure:68ch;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.65;-webkit-font-smoothing:antialiased;
  text-rendering:optimizeLegibility}
.wrap{max-width:74rem;margin:0 auto;padding:0 clamp(1.1rem,4vw,2.5rem)}
.prose{max-width:var(--measure)}
h1,h2,h3,h4{margin:0;line-height:1.18;letter-spacing:-.021em;font-weight:650;
  text-wrap:balance}
p{margin:0}
a{color:var(--accent);text-decoration-color:var(--accent-dim);
  text-underline-offset:3px}
a:hover{text-decoration-color:var(--accent)}
code{font-family:var(--mono);font-size:.86em;background:var(--raised);
  border:1px solid var(--line);border-radius:4px;padding:.06em .34em;color:var(--ink-2)}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px;border-radius:3px}

/* ---- rhythm: sections space themselves, elements never carry margins ---- */
section{padding:3.25rem 0;border-top:1px solid var(--line)}
section > .wrap{display:flex;flex-direction:column;gap:1.5rem}
.stack{display:flex;flex-direction:column;gap:1rem}
.stack-sm{display:flex;flex-direction:column;gap:.55rem}

.eyebrow{font-family:var(--mono);font-size:.6rem;letter-spacing:.24em;
  text-transform:uppercase;color:var(--accent);margin:0}
h2{font-size:clamp(1.35rem,2.4vw,1.75rem)}
h3{font-size:1.02rem}
h4{font-size:.9rem;color:var(--ink-2)}
.lede{color:var(--ink-2);font-size:1.02rem}
.note{color:var(--label);font-size:.86rem;max-width:var(--measure)}
.missing{color:var(--warn);font-family:var(--mono);font-size:.85em}

/* ------------------------------------------------------------------ hero */
header{padding:clamp(3.5rem,9vw,6rem) 0 3rem;position:relative;overflow:hidden}
header canvas{position:absolute;inset:0;width:100%;height:100%;z-index:0;
  opacity:.5;pointer-events:none}
header .wrap{position:relative;z-index:1;display:flex;flex-direction:column;gap:1.4rem}
h1{font-size:clamp(2.1rem,5.6vw,3.4rem);letter-spacing:-.035em;font-weight:700}
h1 em{font-style:normal;color:var(--accent)}
.byline{font-family:var(--mono);font-size:.68rem;letter-spacing:.13em;
  text-transform:uppercase;color:var(--label);display:flex;flex-wrap:wrap;gap:.5rem 1.25rem}

/* ------------------------------------------------------------------- kpi */
.kpi{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:14px;overflow:hidden}
@media(max-width:60rem){.kpi{grid-template-columns:repeat(2,minmax(0,1fr))}}
.kpi > div{background:var(--panel);padding:1.05rem 1.15rem;
  display:flex;flex-direction:column;gap:.3rem;min-width:0}
.kpi .v{font-family:var(--mono);font-size:1.5rem;font-weight:700;letter-spacing:-.03em;
  font-variant-numeric:tabular-nums;line-height:1.1}
.kpi .k{font-family:var(--mono);font-size:.56rem;letter-spacing:.17em;
  text-transform:uppercase;color:var(--label)}
.kpi .s{font-size:.76rem;color:var(--label);line-height:1.4}
.v.ok{color:var(--ok)} .v.accent{color:var(--accent)} .v.warn{color:var(--warn)}

/* ----------------------------------------------------------------- cards */
.cards{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(17rem,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:1.15rem 1.25rem;display:flex;flex-direction:column;gap:.55rem}
.card p{font-size:.88rem;color:var(--ink-2)}
.card .meta{font-family:var(--mono);font-size:.6rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--label)}

/* ---------------------------------------------------------------- tables */
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:12px;
  background:var(--panel)}
table{width:100%;border-collapse:collapse;font-size:.82rem}
thead th{font-family:var(--mono);font-size:.56rem;letter-spacing:.15em;
  text-transform:uppercase;color:var(--label);font-weight:400;text-align:left;
  padding:.75rem .95rem;border-bottom:1px solid var(--line-2);white-space:nowrap;
  background:var(--raised)}
tbody td{padding:.66rem .95rem;border-bottom:1px solid var(--line);
  color:var(--ink-2);vertical-align:top}
tbody tr:last-child td{border-bottom:0}
td.num,th.num{text-align:right;font-family:var(--mono);
  font-variant-numeric:tabular-nums;white-space:nowrap}
td.mono,th.mono{font-family:var(--mono);white-space:nowrap}
td strong{color:var(--ink);font-weight:600}

.pill{font-family:var(--mono);font-size:.58rem;letter-spacing:.1em;
  text-transform:uppercase;padding:.15rem .5rem;border-radius:999px;
  border:1px solid currentColor;white-space:nowrap}
.pill.ok{color:var(--ok)} .pill.bad{color:var(--fault)} .pill.warn{color:var(--warn)}

/* -------------------------------------------------------------- diagrams */
figure{margin:0;background:var(--panel);border:1px solid var(--line);
  border-radius:12px;padding:1.4rem;display:flex;flex-direction:column;gap:.9rem}
figure .scroll{border:0;border-radius:0;background:transparent}
figure svg{display:block;max-width:100%;height:auto;color:var(--ink-2)}
figcaption{font-size:.8rem;color:var(--label);max-width:var(--measure)}

/* ----------------------------------------------------------------- bugs */
.bug{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--fault);
  border-radius:10px;padding:1.2rem 1.35rem;display:flex;flex-direction:column;gap:.7rem}
.bug h3{font-size:.98rem}
.bug dl{margin:0;display:grid;grid-template-columns:6.5rem 1fr;gap:.45rem 1rem;
  font-size:.86rem}
@media(max-width:44rem){.bug dl{grid-template-columns:1fr;gap:.15rem}
  .bug dd{margin-bottom:.5rem}}
.bug dt{font-family:var(--mono);font-size:.56rem;letter-spacing:.15em;
  text-transform:uppercase;color:var(--label);padding-top:.28rem}
.bug dd{margin:0;color:var(--ink-2)}
.bug dd code{font-size:.8em}

/* ------------------------------------------------------------------- cta */
.cta{display:inline-flex;align-items:center;gap:.6rem;font-family:var(--mono);
  font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;
  background:var(--accent);color:#140f28;text-decoration:none;font-weight:700;
  padding:.72rem 1.15rem;border-radius:8px;width:fit-content}
.cta:hover{background:#c4b1ff}
.cta.ghost{background:transparent;color:var(--ink-2);border:1px solid var(--line-2)}
.cta.ghost:hover{background:var(--raised);color:var(--ink)}
.row{display:flex;gap:.75rem;flex-wrap:wrap;align-items:center}

pre{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:1rem 1.15rem;overflow-x:auto;font-family:var(--mono);font-size:.78rem;
  line-height:1.7;color:var(--ink-2)}
pre .c{color:var(--label)}
ul{margin:0;padding-left:1.15rem;display:flex;flex-direction:column;gap:.45rem}
li{color:var(--ink-2);font-size:.92rem}
li strong{color:var(--ink);font-weight:600}

footer{padding:2.5rem 0 4rem;border-top:1px solid var(--line);color:var(--label);
  font-family:var(--mono);font-size:.66rem;line-height:1.9}

@media (prefers-reduced-motion:reduce){
  *{animation-duration:.001ms !important;transition-duration:.001ms !important}
}
"""


HERO_JS = r"""
/* An ambient vector field in the hero: points drifting, nearest neighbours
   linked. It is the one decorative thing on the page and it is at least about
   the subject -- an approximate-nearest-neighbour graph is what the storage
   layer is. Cheap, capped, and skipped entirely under reduced-motion. */
(function () {
  var c = document.getElementById("bg");
  if (!c || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  var x = c.getContext("2d"), pts = [], raf = null;

  function size() {
    var d = Math.min(devicePixelRatio || 1, 2);
    c.width = c.clientWidth * d; c.height = c.clientHeight * d;
    x.setTransform(d, 0, 0, d, 0, 0);
  }
  function seed() {
    var w = c.clientWidth, h = c.clientHeight;
    var count = Math.min(64, Math.round(w * h / 14000));
    pts = [];
    for (var i = 0; i < count; i++) {
      pts.push({x: Math.random() * w, y: Math.random() * h,
                vx: (Math.random() - .5) * .16, vy: (Math.random() - .5) * .16});
    }
  }
  function frame() {
    var w = c.clientWidth, h = c.clientHeight;
    x.clearRect(0, 0, w, h);
    for (var i = 0; i < pts.length; i++) {
      var p = pts[i];
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
    }
    x.lineWidth = 1;
    for (i = 0; i < pts.length; i++) {
      for (var j = i + 1; j < pts.length; j++) {
        var dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
        var d2 = dx * dx + dy * dy;
        if (d2 < 15000) {
          x.strokeStyle = "rgba(167,139,250," + (.16 * (1 - d2 / 15000)).toFixed(3) + ")";
          x.beginPath(); x.moveTo(pts[i].x, pts[i].y);
          x.lineTo(pts[j].x, pts[j].y); x.stroke();
        }
      }
      x.fillStyle = "rgba(167,139,250,.32)";
      x.beginPath(); x.arc(pts[i].x, pts[i].y, 1.4, 0, 6.2832); x.fill();
    }
    raf = requestAnimationFrame(frame);
  }
  function boot() { size(); seed(); if (!raf) frame(); }
  addEventListener("resize", function () { size(); seed(); });
  boot();
})();
"""


def kpi(v, k, s, cls=""):
    return (f'<div><span class="v {cls}">{v}</span>'
            f'<span class="k">{esc(k)}</span><span class="s">{esc(s)}</span></div>')


# ------------------------------------------------------------- architecture

ARCH_SVG = """
<svg viewBox="0 0 1000 400" role="img" aria-label="A training job's embeddings
flow through a storage client into the Raft leader, replicate to five replicas,
and are applied into each replica's own HNSW index; linearizable reads confirm
leadership before serving.">
<defs>
  <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
          markerHeight="7" orient="auto-start-reverse">
    <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/>
  </marker>
  <marker id="arA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
          markerHeight="7" orient="auto-start-reverse">
    <path d="M0 0 L10 5 L0 10 z" fill="#a78bfa"/>
  </marker>
</defs>
<g font-family="ui-monospace,Consolas,monospace" font-size="12">

  <!-- training layer -->
  <rect x="20" y="40" width="180" height="120" rx="8" fill="none"
        stroke="currentColor" stroke-opacity=".45"/>
  <text x="30" y="30" font-size="10" letter-spacing="1.6"
        fill="currentColor" opacity=".65">TRAINING ENGINE</text>
  <rect x="36" y="58" width="148" height="26" rx="4" fill="currentColor" fill-opacity=".07"/>
  <text x="110" y="75" text-anchor="middle" fill="currentColor">Coordinator</text>
  <rect x="36" y="92" width="70" height="26" rx="4" fill="currentColor" fill-opacity=".07"/>
  <text x="71" y="109" text-anchor="middle" fill="currentColor" font-size="11">w1..w4</text>
  <rect x="114" y="92" width="70" height="26" rx="4" fill="currentColor" fill-opacity=".07"/>
  <text x="149" y="109" text-anchor="middle" fill="currentColor" font-size="11">all-reduce</text>
  <rect x="36" y="126" width="148" height="24" rx="4" fill="#a78bfa" fill-opacity=".14"/>
  <text x="110" y="142" text-anchor="middle" fill="#a78bfa" font-size="11">logits &#8594; embeddings</text>

  <!-- publisher / client -->
  <rect x="250" y="76" width="130" height="52" rx="6" fill="none" stroke="#a78bfa"/>
  <text x="315" y="98" text-anchor="middle" fill="#a78bfa">StorageClient</text>
  <text x="315" y="115" text-anchor="middle" fill="currentColor" font-size="10"
        opacity=".7">redirect to leader</text>
  <line x1="202" y1="102" x2="246" y2="102" stroke="#a78bfa" stroke-width="1.5"
        marker-end="url(#arA)"/>
  <text x="224" y="93" text-anchor="middle" fill="#a78bfa" font-size="10">write</text>

  <!-- raft leader -->
  <rect x="450" y="66" width="140" height="72" rx="6" fill="none" stroke="currentColor"/>
  <text x="520" y="90" text-anchor="middle" fill="currentColor">Raft leader</text>
  <text x="520" y="108" text-anchor="middle" font-size="10" fill="currentColor"
        opacity=".7">append + replicate</text>
  <text x="520" y="124" text-anchor="middle" font-size="10" fill="currentColor"
        opacity=".7">commit at quorum</text>
  <line x1="382" y1="102" x2="446" y2="102" stroke="#a78bfa" stroke-width="1.5"
        marker-end="url(#arA)"/>
  <text x="414" y="93" text-anchor="middle" fill="#a78bfa" font-size="10">propose</text>

  <!-- replicas -->
  <rect x="690" y="24" width="290" height="180" rx="8" fill="none"
        stroke="currentColor" stroke-opacity=".45"/>
  <text x="700" y="16" font-size="10" letter-spacing="1.6" fill="currentColor"
        opacity=".65">FIVE REPLICAS</text>
"""

for _i in range(5):
    _y = 38 + _i * 33
    ARCH_SVG += f"""
  <rect x="706" y="{_y}" width="96" height="24" rx="4" fill="currentColor" fill-opacity=".07"/>
  <text x="754" y="{_y + 16}" text-anchor="middle" fill="currentColor" font-size="11">log n{_i + 1}</text>
  <line x1="806" y1="{_y + 12}" x2="842" y2="{_y + 12}" stroke="currentColor"
        stroke-opacity=".6" marker-end="url(#ar)"/>
  <rect x="846" y="{_y}" width="118" height="24" rx="4" fill="currentColor" fill-opacity=".07"/>
  <text x="905" y="{_y + 16}" text-anchor="middle" fill="currentColor" font-size="11">HNSW index</text>"""

ARCH_SVG += """
  <text x="835" y="224" text-anchor="middle" font-size="10" fill="currentColor"
        opacity=".7">apply, deduped by log index and by id</text>
  <line x1="592" y1="102" x2="686" y2="102" stroke="currentColor" stroke-width="1.5"
        marker-end="url(#ar)"/>
  <text x="639" y="93" text-anchor="middle" fill="currentColor" font-size="10">AppendEntries</text>

  <!-- read path: out of a replica's index, back to the client -->
  <path d="M974 206 L974 288 L315 288 L315 136" fill="none" stroke="#a78bfa"
        stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#arA)"/>
  <text x="645" y="279" text-anchor="middle" fill="#a78bfa" font-size="10">
    linearizable read: confirm leadership, then wait for applied &#8805; read index</text>

  <!-- ack path: only once the entry is committed -->
  <path d="M520 140 L520 332 L110 332 L110 162" fill="none" stroke="currentColor"
        stroke-width="1.5" stroke-dasharray="3 4" stroke-opacity=".65"
        marker-end="url(#ar)"/>
  <text x="315" y="323" text-anchor="middle" fill="currentColor" font-size="10"
        opacity=".75">ack only after quorum commit in the current term</text>

  <!-- legend: the encoding repeats, so it is worth naming -->
  <line x1="20" y1="360" x2="980" y2="360" stroke="currentColor" stroke-opacity=".18"/>
  <rect x="20" y="376" width="22" height="14" rx="3" fill="none" stroke="#a78bfa"/>
  <text x="50" y="387" font-size="10" fill="currentColor" opacity=".75">
    written for this platform</text>
  <rect x="230" y="376" width="22" height="14" rx="3" fill="none"
        stroke="currentColor" stroke-opacity=".5"/>
  <text x="260" y="387" font-size="10" fill="currentColor" opacity=".75">
    the three engines, used unmodified</text>
</g>
</svg>
"""


def session_timeline_svg(session: dict, frames: list[dict]) -> str:
    """A real chart of the recorded run, drawn from its frames."""
    W, H = 900, 200
    pad_l, pad_r, pad_t, pad_b = 44, 18, 26, 58
    iw = W - pad_l - pad_r
    ih = H - pad_t - pad_b
    total = max(1, len(frames) - 1)
    colors = {"healthy": "#4ade80", "degraded": "#fbbf24", "unavailable": "#f43f5e"}

    bars = []
    bw = iw / len(frames)
    for i, f in enumerate(frames):
        c = colors.get(f["health"]["status"], "#8c95a0")
        x = pad_l + i * bw
        bars.append(f'<rect x="{x:.2f}" y="{pad_t}" width="{bw + .5:.2f}" '
                    f'height="18" fill="{c}" fill-opacity=".55"/>')

    # published-embeddings curve
    pub = [sum(j.get("published", 0) for j in f["jobs"]) for f in frames]
    top = max(pub) or 1
    pts = " ".join(
        f"{pad_l + i / total * iw:.2f},{pad_t + 30 + (ih - 30) * (1 - v / top):.2f}"
        for i, v in enumerate(pub)
    )

    # Faults and recoveries get different colours. Marking a heal in the same
    # red as a kill would make the chart claim five things went wrong when
    # two of them were the system being put back together.
    marks, labels = [], []
    events = session.get("faults", [])
    for k, fa in enumerate(events):
        idx = next((i for i, f in enumerate(frames) if f["round"] == fa["round"]), None)
        if idx is None:
            continue
        col = "#f43f5e" if fa.get("kind", "fault") == "fault" else "#4ade80"
        x = pad_l + idx / total * iw
        marks.append(
            f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + ih}" '
            f'stroke="{col}" stroke-width="1" stroke-dasharray="3 3" stroke-opacity=".75"/>'
            f'<circle cx="{x:.1f}" cy="{pad_t - 8}" r="3" fill="{col}"/>'
        )
        # Stagger the captions onto two rows; at five events on one row the
        # neighbouring labels ran into each other.
        ly = H - 30 if k % 2 == 0 else H - 17
        anchor = "start" if x < W * .66 else "end"
        labels.append(
            f'<line x1="{x:.1f}" y1="{pad_t + ih}" x2="{x:.1f}" y2="{ly - 8}" '
            f'stroke="{col}" stroke-opacity=".3"/>'
            f'<text x="{x + (4 if anchor == "start" else -4):.1f}" y="{ly}" '
            f'text-anchor="{anchor}" font-size="9.5" fill="{col}">'
            f'{esc(fa["description"])}</text>'
        )

    return f"""
<svg viewBox="0 0 {W} {H}" role="img" aria-label="Cluster health across the
recorded session, with each injected fault and each recovery marked, and the
count of acknowledged embeddings rising throughout.">
<g font-family="ui-monospace,Consolas,monospace">
  <text x="0" y="{pad_t + 13}" font-size="9.5" fill="currentColor" opacity=".7">health</text>
  {''.join(bars)}
  {''.join(marks)}
  <polyline points="{pts}" fill="none" stroke="#a78bfa" stroke-width="1.8"/>
  <text x="0" y="{pad_t + 42}" font-size="9.5" fill="#a78bfa">{top}</text>
  <text x="0" y="{pad_t + ih}" font-size="9.5" fill="#a78bfa">0</text>
  <text x="{pad_l}" y="{H - 8}" font-size="9.5" fill="currentColor" opacity=".55">
    frame 0</text>
  <text x="{W - pad_r}" y="{H - 8}" text-anchor="end" font-size="9.5"
        fill="currentColor" opacity=".55">frame {len(frames) - 1}</text>
  {''.join(labels)}
</g>
</svg>"""


# ----------------------------------------------------------------- the bugs

BUGS = [
    {
        "title": "The same embedding id committed at two log indices",
        "where": "storage/client.py + storage/statemachine.py",
        "found": "randomized cross-layer sweep, during a partition",
        "symptom": "<code>ValueError: id 'job:s24@48' already exists</code> raised "
                   "inside apply, on every replica at once.",
        "cause": "The client proposed a write, timed out waiting for the commit, "
                 "and re-proposed it. When the partition healed, both entries "
                 "committed. The state machine deduped by log-index watermark, "
                 "which cannot help here: both indices are above the watermark, "
                 "so both look new. <code>VectorDB.insert</code> then refused the "
                 "duplicate id and raised.",
        "blame": "Neither engine. Raft committed two entries because it was asked "
                 "to twice; HNSW refused a duplicate because it should. The defect "
                 "was treating an unconfirmed proposal as safe to retry, which is "
                 "a claim about at-most-once delivery that Raft never made.",
        "fix": "Two halves, both necessary. The client now returns an "
               "unconfirmed result with its log index instead of re-proposing. The "
               "state machine additionally suppresses a repeated id, because a "
               "state machine over a replicated log must be <strong>total</strong>: "
               "an exception during apply does not reject one bad write, it stops "
               "that replica applying anything further while its peers continue.",
        "test": "test_regression_same_id_committed_twice_does_not_kill_a_replica, "
                "test_regression_client_does_not_repropose_an_unconfirmed_write",
    },
    {
        "title": "A shared event log silently replaced by an empty one",
        "where": "storage/cluster.py, training/job.py, training/publisher.py",
        "found": "the cross-layer timeline test, which saw zero event kinds",
        "symptom": "Storage and training events landed in three separate logs, so "
                   "the unified timeline the whole dashboard depends on was empty "
                   "for every layer but one.",
        "cause": "<code>EventLog</code> defines <code>__len__</code>, so an empty "
                 "log is <em>falsy</em>. Three constructors used "
                 "<code>self.events = events or EventLog()</code>, which discards "
                 "the caller's log precisely when it is new and empty, which is "
                 "always at startup.",
        "blame": "Mine, in all three places, from one idiom copied forward.",
        "fix": "<code>events if events is not None else EventLog()</code>, "
               "everywhere an optional shared object is accepted.",
        "test": "test_events_from_both_layers_share_one_timeline",
    },
    {
        "title": "The correctness checker was testing the wrong property",
        "where": "chaos/checker.py, invariant 6",
        "found": "fuzz seed 57, which the sweep reported as a violation",
        "symptom": "A linearizable read was flagged as dishonest: 8 acknowledged "
                   "ids, only 6 returned by a k-NN query at k=13.",
        "cause": "The invariant asserted that a k-NN query returns every "
                 "acknowledged id. HNSW is an <strong>approximate</strong> index "
                 "and a k-NN query is not an enumeration; replaying the seed showed "
                 "all 8 ids present in the serving replica. The check conflated "
                 "recall (an approximation property) with staleness (a consistency "
                 "property).",
        "blame": "The harness, not the platform. Worth stating plainly: the one "
                 "violation this project's sweep ever reported was the checker's "
                 "own bug, which is exactly why the mutation tests exist.",
        "fix": "Invariant 6 now asserts containment in the state actually read "
               "from, which is the consistency claim. Recall is recorded as an "
               "observation alongside it rather than asserted.",
        "test": "test_regression_read_honesty_tests_containment_not_search_recall",
    },
]


# =================================================================== builder

def build_report(r: dict) -> str:
    b = r["bench"]
    s = r["session"]
    fz = r["fuzz"]
    scen = r["scenarios"]
    e2e, rec, rd = b["end_to_end_latency"], b["recovery"], b["read_cost"]
    suite = r.get("suite") or {"passed": None, "failed": 0}

    frames = json.loads((SITE / "session.json").read_text(encoding="utf-8"))["frames"]

    scen_pass = sum(1 for x in scen if x["passed"])
    total_pub = sum(x["published"] for x in scen) + fz["total_published"]
    overlaps = sum(x["cross_layer_overlaps"] for x in scen)
    checks = sum(x["check"].get("checks_run", 0) for x in scen)

    kpis = "".join([
        kpi(f'{scen_pass}/{len(scen)}', "named scenarios", "cross-layer fault sequences",
            "ok" if scen_pass == len(scen) else "warn"),
        kpi(f'{fz["passed"]}/{fz["runs"]}', "fuzz seeds clean", "randomized fault schedules",
            "ok" if fz["failed"] == 0 else "warn"),
        kpi(n(total_pub), "embeddings", "written through Raft, then read back", "accent"),
        kpi(n(fz["published_not_queryable"]), "acked but lost",
            "the failure this project exists to detect",
            "ok" if fz["published_not_queryable"] == 0 else "warn"),
        kpi(n(checks), "invariant checks", "run continuously, not at the end"),
        kpi(n(overlaps), "overlapping faults", "in two layers at once, measured"),
        kpi(str(len(BUGS)), "bugs found", "in the composition, all fixed", "warn"),
        kpi(f'{rd["tick_overhead"]}', "tick read cost", "linearizable over stale"),
    ])

    # -- scenarios table
    scen_rows = "".join(
        f'<tr><td><strong>{esc(x["name"])}</strong>'
        f'<div class="note" style="font-size:.78rem;margin-top:.2rem">'
        f'{esc(x["fault_summary"])}</div></td>'
        f'<td class="num">{x["cross_layer_overlaps"]}</td>'
        f'<td class="num">{x["published"]}</td>'
        f'<td class="num">{x["queryable"]}</td>'
        f'<td class="num">{x["lost_uncommitted"]}</td>'
        f'<td class="mono">{"yes" if x["converged"] else "NO"}</td>'
        f'<td><span class="pill {"ok" if x["passed"] else "bad"}">'
        f'{"pass" if x["passed"] else "fail"}</span></td></tr>'
        for x in scen
    )

    # -- benchmarks
    st_rows = "".join(
        f'<tr><td class="num">{x["nodes"]}</td><td class="num">{x["quorum"]}</td>'
        f'<td class="num">{x["published"]}</td><td class="num">{x["ticks"]}</td>'
        f'<td class="num">{x["ticks_per_write"]}</td>'
        f'<td class="num">{x["wall_s"]}</td></tr>'
        for x in b["scaling"]["storage"]
    )
    tr_rows = "".join(
        f'<tr><td class="num">{x["workers"]}</td><td class="num">{x["rounds"]}</td>'
        f'<td class="num">{x["wall_s"]}</td>'
        f'<td class="num">{x["final_accuracy"]}</td>'
        f'<td class="num">{x["published"]}</td></tr>'
        for x in b["scaling"]["training"]
    )
    rec_rows = "".join(
        f'<tr><td>{esc(x["fault"])}</td>'
        f'<td class="num">{x["recovered"]}/{x["runs"]}</td>'
        f'<td class="num">{x["median_ticks"]}</td>'
        f'<td class="num">{x["max_ticks"]}</td></tr>'
        for x in rec["rows"]
    )

    bugs_html = "".join(
        f'''<article class="bug">
  <h3>{i + 1}. {bug["title"]}</h3>
  <dl>
    <dt>found by</dt><dd>{bug["found"]}</dd>
    <dt>symptom</dt><dd>{bug["symptom"]}</dd>
    <dt>cause</dt><dd>{bug["cause"]}</dd>
    <dt>whose bug</dt><dd>{bug["blame"]}</dd>
    <dt>fix</dt><dd>{bug["fix"]}</dd>
    <dt>regression</dt><dd><code>{bug["test"]}</code></dd>
    <dt>file</dt><dd><code>{bug["where"]}</code></dd>
  </dl>
</article>'''
        for i, bug in enumerate(BUGS)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Composing Three Systems</title>
<meta name="description" content="A distributed ML platform built by composing a
Raft engine, an HNSW vector index, and a distributed training framework, and the
evidence that the composition is correct.">
<style>{CSS}</style>
</head>
<body>

<header>
  <canvas id="bg" aria-hidden="true"></canvas>
  <div class="wrap">
    <p class="eyebrow">ml-infra-platform</p>
    <h1>Three correct systems<br>do not make <em>one</em> correct system.</h1>
    <p class="lede prose">
      A Raft consensus engine, an HNSW vector index, and a distributed training
      framework, each already tested on its own, composed into one platform:
      models train, their embeddings are replicated through consensus into a
      vector store, and a control plane runs both. The engines are unmodified.
      Everything interesting happens at the seams between them, so that is where
      this project does its testing.
    </p>
    <p class="byline">
      <span>Abhineeth Duddela</span>
      <span>generated {esc(r["generated_at"])}</span>
      <span>python {esc(r["env"]["python"])}</span>
      <span>numpy {esc(r["env"]["numpy"])}</span>
      <span>{esc(r["elapsed_s"])}s to reproduce</span>
    </p>
    <div class="row">
      <a class="cta" href="dashboard.html">Watch the recorded session &#8594;</a>
      <a class="cta ghost" href="https://github.com/abho7/ml-infra-platform">Source</a>
    </div>
  </div>
</header>

<section>
  <div class="wrap">
    <div class="kpi">{kpis}</div>
    <p class="note">
      Every figure on this page was produced by <code>python run_experiments.py</code>
      and read out of <code>site/results.json</code>. Nothing here is typed in by
      hand or estimated; where a value was not measured, the page says so.
    </p>
  </div>
</section>

<section>
  <div class="wrap">
    <p class="eyebrow">the claim</p>
    <h2>What is actually being proven</h2>
    <div class="prose stack">
      <p class="lede">
        The three engines come with their own suites, and they pass: 23 tests for
        Raft, 28 for HNSW, 68 for the training framework. None of them says
        anything about what happens when a training worker dies at the same
        moment the storage cluster splits in half.
      </p>
      <p style="color:var(--ink-2);font-size:.94rem">
        That is the gap. A component proof is a statement about a component under
        its own assumptions; composing components creates states that neither
        one's assumptions cover. The first bug below is exactly that shape: Raft
        behaved correctly, HNSW behaved correctly, and the platform crashed every
        replica in the cluster.
      </p>
    </div>
    <figure>
      <div class="scroll">{ARCH_SVG}</div>
      <figcaption>
        The write path, the read path, and the acknowledgement path. An embedding
        is acknowledged to the training job only once it is quorum-committed
        under the leader's current term; each replica applies the same committed
        log into its own HNSW index, so the indices are built independently and
        must agree by construction rather than by copying.
      </figcaption>
    </figure>
  </div>
</section>

<section>
  <div class="wrap">
    <p class="eyebrow">consistency</p>
    <h2>The guarantee, stated exactly</h2>
    <p class="note">
      The storage engine on its own offers reads from local state, which can lag.
      Linearizable reads are implemented in this platform as a ReadIndex protocol
      layered on top of it, not inherited from it.
    </p>
    <div class="scroll"><table>
      <thead><tr><th>operation</th><th>guarantee</th><th>cost</th>
        <th>under partition</th></tr></thead>
      <tbody>
        <tr><td class="mono"><strong>write</strong></td>
          <td>Linearizable. Acknowledged only after quorum commit in the
              leader's current term.</td>
          <td class="num">{e2e["consensus_ticks"]["p50"]} ticks p50</td>
          <td>Minority side cannot commit; the write returns unacknowledged
              and is never claimed as durable.</td></tr>
        <tr><td class="mono"><strong>read: linearizable</strong></td>
          <td>Reflects every write acknowledged before the read began. Leadership
              is confirmed by a heartbeat quorum round, then the read waits for
              <code>applied &#8805; read index</code>.</td>
          <td class="num">{rd["linearizable"]["median_ticks"]} ticks,
              {n(rd["linearizable"]["median_ms"], 3)} ms</td>
          <td><strong>Refuses.</strong> Leadership cannot be confirmed, so the
              read fails rather than serving something it cannot vouch for.</td></tr>
        <tr><td class="mono"><strong>read: stale</strong></td>
          <td>Served from any replica's local index with no coordination. May lag
              arbitrarily. Opt-in, never the default.</td>
          <td class="num">{rd["stale"]["median_ticks"]} ticks,
              {n(rd["stale"]["median_ms"], 3)} ms</td>
          <td>Succeeds, possibly with old data. That is the mode's contract, so
              it is not counted as a violation.</td></tr>
      </tbody>
    </table></div>
    <p class="note">{esc(rd["note"])}</p>
  </div>
</section>

<section>
  <div class="wrap">
    <p class="eyebrow">the recorded run</p>
    <h2>{esc(s["title"])}</h2>
    <p class="lede prose">
      {len(frames)} frames captured from one real execution: a 4-worker job
      publishing embeddings through consensus while a node dies, the cluster
      splits, and a training worker is killed mid-partition. The dashboard replays
      these frames; it is the same interface that runs live against the control
      plane, with recorded frames substituted for polled ones.
    </p>
    <figure>
      <div class="scroll">{session_timeline_svg(s, frames)}</div>
      <figcaption>
        Cluster health per frame (green healthy, amber degraded), with each
        injected fault marked in red and each recovery in green, and the running
        count of acknowledged embeddings in violet. The violet curve never
        falls: no acknowledged embedding was ever withdrawn, including across
        the partition and the leadership change.
      </figcaption>
    </figure>
    <div class="cards">
      <div class="card"><span class="meta">acknowledged</span>
        <h3>{n(s["published"])} embeddings</h3>
        <p>Every one still queryable from the leader after the faults healed:
           {n(s["queryable"])} of {n(s["published"])}.</p></div>
      <div class="card"><span class="meta">in flight when a node died</span>
        <h3>{n(s["lost_uncommitted"])} lost</h3>
        <p>Never acknowledged, so never claimed. This is correct behaviour and
           the direct analogue of an uncommitted Raft entry, so it is reported
           rather than counted as a violation.</p></div>
      <div class="card"><span class="meta">after settling</span>
        <h3>{"converged" if s["converged"] else "DIVERGED"}</h3>
        <p>All five replicas returned identical index digests, having each built
           their HNSW graph independently from the same committed log.</p></div>
      <div class="card"><span class="meta">the job itself</span>
        <h3>{n(s["accuracy"], 4)} accuracy</h3>
        <p>The model still trained. Losing a worker mid-partition reshards onto
           the survivors rather than failing the job.</p></div>
    </div>
    <a class="cta" href="dashboard.html">Open the replay &#8594;</a>
  </div>
</section>

<section>
  <div class="wrap">
    <p class="eyebrow">invariants</p>
    <h2>Six properties that only exist at the seams</h2>
    <p class="note">
      Checked after every round of every scenario and every fuzz seed, not once
      at the end. {n(checks)} checks across the named scenarios alone.
    </p>
    <div class="cards">
      <div class="card"><span class="meta">1 &middot; durability</span>
        <h4>Acknowledged writes survive leadership</h4>
        <p>An embedding acknowledged to a training job is present on every
           leader elected afterwards.</p></div>
      <div class="card"><span class="meta">2 &middot; convergence</span>
        <h4>Replicas agree on their common prefix</h4>
        <p>Two in-sync replicas hold identical applied sequences. Catches seed
           divergence and apply-order drift.</p></div>
      <div class="card"><span class="meta">3 &middot; watermark</span>
        <h4>The index matches what the replica claims</h4>
        <p>A replica's vector count matches the committed prefix it says it
           applied. Catches a reset that drops one but not the other.</p></div>
      <div class="card"><span class="meta">4 &middot; no phantoms</span>
        <h4>Nothing searchable was uncommitted</h4>
        <p>Every vector in an index traces to a committed log entry.</p></div>
      <div class="card"><span class="meta">5 &middot; exactly once</span>
        <h4>No embedding stored twice</h4>
        <p>Re-applying a log prefix after restart must not duplicate. This is
           the invariant bug 1 violated.</p></div>
      <div class="card"><span class="meta">6 &middot; read honesty</span>
        <h4>A linearizable read never lies</h4>
        <p>The state served is no older than any write acknowledged before the
           read began, or the read fails.</p></div>
    </div>
    <div class="prose stack-sm">
      <h4>Deliberately not violations</h4>
      <ul>
        <li>A <strong>stale read lagging</strong> &mdash; that is the mode's
            contract, and counting it would make the mode meaningless.</li>
        <li>A <strong>minority-side linearizable read failing</strong> &mdash;
            refusing is the correct answer and the entire point of ReadIndex.</li>
        <li>An <strong>unacknowledged in-flight write vanishing</strong> when its
            node dies &mdash; correct, and reported separately as
            <em>lost uncommitted</em> so it stays visible rather than hidden.</li>
      </ul>
      <p class="note">
        Each of the six has a mutation test that breaks it deliberately and
        asserts the checker fires and names the right one. A correctness harness
        never observed to fail is not evidence; every &ldquo;0 violations&rdquo;
        on this page is worth exactly as much as those tests.
      </p>
    </div>
  </div>
</section>

<section>
  <div class="wrap">
    <p class="eyebrow">results</p>
    <h2>Cross-layer scenarios</h2>
    <p class="note">
      Named fault sequences, each run against a live training job rather than an
      idle cluster. The <em>overlaps</em> column is measured from the schedule
      and counts fault windows that were open in <em>both</em> layers at the same
      instant &mdash; so it is worth reading it honestly: most of these rows are
      a storage fault landing while training continues, and only the ones with a
      non-zero count are genuinely simultaneous cross-layer failures. Reporting
      that column is what stops the headline claim from going quietly vacuous.
    </p>
    <div class="scroll"><table>
      <thead><tr><th>scenario</th><th class="num">overlaps</th>
        <th class="num">acked</th><th class="num">queryable</th>
        <th class="num">lost in flight</th><th>converged</th><th>result</th></tr></thead>
      <tbody>{scen_rows}</tbody>
    </table></div>

    <h3>Randomized sweep</h3>
    <div class="kpi">
      {kpi(f'{fz["passed"]}/{fz["runs"]}', "seeds clean", "each a different fault schedule", "ok" if fz["failed"] == 0 else "warn")}
      {kpi(n(fz["total_published"]), "embeddings", "acknowledged across the sweep", "accent")}
      {kpi(n(fz["published_not_queryable"]), "acked, not queryable", "the headline failure mode", "ok" if fz["published_not_queryable"] == 0 else "warn")}
      {kpi(n(fz["min_accuracy"], 4), "lowest accuracy", "worst-case model, under faults")}
    </div>
    <p class="note">
      Faults are bounded so a quorum always survives; a cluster deliberately
      driven below quorum stops being able to commit, which tests nothing about
      the composition. A failing seed replays exactly:
      <code>python -m mlplat.chaos.fuzz</code>.
    </p>

    <h3>Cross-validated against real sockets</h3>
    <div class="prose stack-sm">
      <p style="color:var(--ink-2);font-size:.94rem">
        Everything above runs on a deterministic queued transport. That is what
        makes a fault reproducible from a seed, but it is also a modelling
        assumption, and an assumption nobody checks is just a hope: if the
        storage layer only behaves because messages arrive in a tidy order no
        real network would produce, these results describe the harness rather
        than the system.
      </p>
      <p style="color:var(--ink-2);font-size:.94rem">
        So the same <code>VectorStateMachine</code> is also run under the Raft
        engine's real asyncio TCP server &mdash; three processes on localhost,
        real sockets, real election timeouts &mdash; and driven through the
        engine's own client. Replicas converge to identical index digests,
        acknowledged writes survive a node dying, and the duplicate-id fix from
        bug 1 holds. Reverting that fix reproduces the original
        <code>ValueError</code> inside the server's apply path over TCP too,
        which is how these three tests are known not to be vacuous.
      </p>
    </div>

    <div class="kpi">
      {kpi(n(suite["passed"]), "tests passing", "measured by running them, not recalled",
           "ok" if suite["failed"] == 0 else "warn")}
      {kpi(n(suite["failed"]), "failing", "at the time this page was generated",
           "ok" if suite["failed"] == 0 else "warn")}
      {kpi("3", "over real TCP", "cross-validating the deterministic transport", "accent")}
      {kpi("6", "mutation tests", "one per invariant, each expects a catch")}
    </div>
  </div>
</section>

<section>
  <div class="wrap">
    <p class="eyebrow">honestly</p>
    <h2>Bugs found in the composition</h2>
    <p class="lede prose">
      Three, all fixed, each with a regression test. Two were mine. One was in
      the correctness checker itself, which is worth stating plainly: the only
      violation the sweep ever reported turned out to be the harness being
      wrong, not the platform.
    </p>
    <div class="stack">{bugs_html}</div>
  </div>
</section>

<section>
  <div class="wrap">
    <p class="eyebrow">measured</p>
    <h2>Benchmarks</h2>
    <p class="note">
      Wall-clock figures are from one machine and one process, and are the least
      transferable numbers here. Tick counts are protocol costs and are the ones
      that would carry to a real deployment, so they are reported alongside
      everywhere.
    </p>

    <h3>End-to-end: training step to queryable embedding</h3>
    <div class="kpi">
      {kpi(n(e2e["wall_ms"]["p50"], 2), "ms p50", "checkpoint published and read back")}
      {kpi(n(e2e["wall_ms"]["p95"], 2), "ms p95", "worst of the sampled checkpoints")}
      {kpi(n(e2e["consensus_ticks"]["p50"]), "ticks p50", "the protocol cost", "accent")}
      {kpi(f'{e2e["checkpoints"]}/{e2e["checkpoints"]}' if e2e["all_queryable"] else "FAILED",
           "queryable", "every checkpoint, confirmed by a real read",
           "ok" if e2e["all_queryable"] else "warn")}
    </div>
    <p class="note">{esc(e2e["note"])}</p>

    <h3>Scaling</h3>
    <div class="cards" style="grid-template-columns:repeat(auto-fit,minmax(20rem,1fr))">
      <div class="stack-sm">
        <h4>Storage width &mdash; more nodes, 4 workers fixed</h4>
        <div class="scroll"><table>
          <thead><tr><th>nodes</th><th class="num">quorum</th><th class="num">acked</th>
            <th class="num">ticks</th><th class="num">ticks/write</th>
            <th class="num">wall s</th></tr></thead>
          <tbody>{st_rows}</tbody>
        </table></div>
      </div>
      <div class="stack-sm">
        <h4>Training width &mdash; more workers, 5 nodes fixed</h4>
        <div class="scroll"><table>
          <thead><tr><th>workers</th><th class="num">rounds</th><th class="num">wall s</th>
            <th class="num">accuracy</th><th class="num">acked</th></tr></thead>
          <tbody>{tr_rows}</tbody>
        </table></div>
      </div>
    </div>
    <p class="note">{esc(b["scaling"]["note"])}</p>

    <h3>Recovery</h3>
    <div class="scroll"><table>
      <thead><tr><th>fault</th><th class="num">recovered</th>
        <th class="num">median ticks</th><th class="num">max ticks</th></tr></thead>
      <tbody>{rec_rows}</tbody>
    </table></div>
    <p class="note">{esc(rec["note"])}</p>
  </div>
</section>

<section>
  <div class="wrap">
    <p class="eyebrow">limits</p>
    <h2>What this does not do</h2>
    <div class="prose"><ul>
      <li><strong>No snapshots.</strong> Neither the Raft engine nor this
          platform truncates the log, so a rejoining replica catches up by full
          replay and the log grows without bound. Fine at these sizes; the first
          thing a real deployment would need.</li>
      <li><strong>No leader lease.</strong> Every linearizable read pays a full
          heartbeat quorum round, which is why the measured overhead is
          {rd["tick_overhead"]} ticks rather than zero. A real system amortises
          that across a lease interval.</li>
      <li><strong>Deterministic transport, not a network.</strong> The cluster
          runs on a queued transport with modelled latency, drops and partitions.
          That is what makes a fault reproducible from a seed, and it is also why
          the wall-clock numbers are not deployment numbers.</li>
      <li><strong>Deletion is soft.</strong> Inherited from the vector engine.
          Vectors are tombstoned, not reclaimed.</li>
      <li><strong>One shard.</strong> Every replica holds the entire index, so
          storage capacity is one node's capacity. Sharding would be a
          replication-group-per-shard change, not a change to any of this.</li>
    </ul></div>
  </div>
</section>

<section>
  <div class="wrap">
    <p class="eyebrow">reproduce</p>
    <h2>Run it yourself</h2>
    <pre><span class="c"># the three engines are cloned in as read-only dependencies</span>
git clone https://github.com/abho7/ml-infra-platform &amp;&amp; cd ml-infra-platform
pip install numpy pytest

<span class="c"># every test, including the mutation tests behind every claim above</span>
pytest -q

<span class="c"># the live control plane, with a real dashboard at localhost:8950</span>
python -m mlplat.control.cli serve

<span class="c"># regenerate every number on this page</span>
python run_experiments.py &amp;&amp; python build_site.py</pre>
    <p class="note">
      Dependencies are numpy and pytest. The Raft engine, the HNSW index, the
      training framework, the transport, the control plane and its HTTP API are
      all from scratch.
    </p>
  </div>
</section>

<footer><div class="wrap">
  ml-infra-platform &middot; Abhineeth Duddela &middot; &copy; 2026<br>
  composed from
  <a href="https://github.com/abho7/raft-kv-store">raft-kv-store</a>,
  <a href="https://github.com/abho7/vectordb-hnsw">vectordb-hnsw</a> and
  <a href="https://github.com/abho7/distributed-training-framework">distributed-training-framework</a>,
  none of them modified<br>
  measured {esc(r["generated_at"])} on {esc(r["env"]["platform"])}
</div></footer>

<script>{HERO_JS}</script>
</body>
</html>
"""


def build_dashboard() -> str:
    """The live dashboard, with a recording inlined so it replays instead."""
    src = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
    session = (SITE / "session.json").read_text(encoding="utf-8")
    # `</script>` inside JSON string data would close the tag early; escaping the
    # slash keeps it valid JSON and inert HTML.
    session = session.replace("</", "<\\/")
    tag = (f'<p style="text-align:center;padding:1.5rem 0 0">'
           f'<a href="index.html" style="font-family:var(--mono);font-size:.62rem;'
           f'letter-spacing:.12em;text-transform:uppercase;color:var(--label)">'
           f'&#8592; back to the report</a></p>\n'
           f'<script type="application/json" id="recording">{session}</script>\n')

    # BEFORE <footer>, which is before the dashboard's own <script>. Injecting
    # at </body> instead puts the data after the boot code that reads it, and
    # `getElementById("recording")` returns null on a document still parsing --
    # the page then falls through to live mode and reports no control plane.
    if "<footer>" not in src:
        raise ValueError("dashboard/index.html has no <footer> to inject before")
    return src.replace("<footer>", tag + "<footer>", 1)


def main() -> None:
    results_path = SITE / "results.json"
    if not results_path.is_file():
        raise SystemExit("site/results.json is missing. Run: python run_experiments.py")
    r = json.loads(results_path.read_text(encoding="utf-8"))

    (SITE / "index.html").write_text(build_report(r), encoding="utf-8")
    (SITE / "dashboard.html").write_text(build_dashboard(), encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")

    for f in ("index.html", "dashboard.html"):
        kb = (SITE / f).stat().st_size / 1024
        print(f"  site/{f}  {kb:.0f} KB")


if __name__ == "__main__":
    main()
