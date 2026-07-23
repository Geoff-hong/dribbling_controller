"""Per-condition run identity, so a changed condition table can be topped up
instead of re-run from scratch.

The old guard hashed the WHOLE table: add one axis and every episode already on
disk was declared incompatible. Here each condition carries its own fingerprint,
so a run splits into

    reusable  -- fingerprint unchanged, episodes on disk stay
    changed   -- same name, different semantics; its rows must be DROPPED
    new       -- absent from the previous run
    stale     -- in the previous run but no longer in the table

and only `changed` + `new` are executed.

Three couplings had to be broken for "reusable" to actually mean unchanged:

* the per-episode RNG used to be seeded by the condition's POSITION in the table
  (`_seed_index = enumerate(table)`), so inserting an axis shifted the stream of
  every condition after it. `seed_index()` below hashes the condition NAME
  instead — stable under insertion, reordering, and sharding alike.
* `--seed` / `--route-bank` / `--episode-s` / `--standby-hold-s` were in no
  fingerprint at all, so resuming with a different one silently mixed protocols.
  They are folded into every condition fingerprint now. `--reps` deliberately is
  NOT: more reps are just more episodes of the same condition, which the existing
  resume already handles, so raising it is itself a free top-up.
* the engine's own source was never recorded. See code_fingerprint().

Mixing episodes recorded at different times is statistically sound -- which robot
slot an episode lands on is effectively random either way, so rates stay
unbiased -- but it is NOT bit-reproducible. That is the price of topping up.
"""
import hashlib
import json
import os
import subprocess

# Fields of a condition dict that do not affect what is simulated.
_COSMETIC = ("name", "group", "axis", "_seed_index")

# Source files whose content can change simulated behaviour.
_CODE_FILES = ("engine.py", "runner.py", "conditions.py")

# Bumped whenever a change makes new episodes statistically incomparable to old
# ones on the SAME condition name. Recorded per run and surfaced by the report,
# because top-up reuse and cross-run comparison are both silently wrong across a
# protocol change.
#   1 -> pre-2026-07-22
#   2 -> robot DR off at nominal (engine.TRAIN_DR), body-ground contact restored,
#        mj_setConst after per-episode model edits, per-joint actuator gains,
#        payload inertia recompute, U(0,dv) ball push, off-route dwell + gate
#   3 -> nominal synthetic latency 2 steps / 10 ms -> C++-parity 0 / 0;
#        strict success truly nests inside route/possession; human routes use
#        the C++ lazy generator and exact mt19937 float stream (no Python-only
#        clearance redraw/governor, and equal seeds now mean equal routes)
#   4 -> full C++ sim2sim timing parity: bridge_delay_ms models the 100 Hz
#        topic hop (ball + base obs, obs frame and route input one publish =
#        10 ms stale; joints fresh; action same-step); default --episode-s
#        15 s -> 12 s
PROTOCOL_VERSION = 4


def _sha(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=list).encode()).hexdigest()


def file_sha(path, blocks=()):
    """Content hash of a file plus any sidecars, None when it does not exist.

    Recording a BASENAME is not provenance: every checkpoint dir here ships its
    policy as `softtouch_dribble_deploy.onnx`, so six different policies share
    one name and a resume could not tell them apart. Weights also live in a
    separate `.onnx.data` sidecar that the graph file's hash does not cover.
    """
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha1()
    for p in (path,) + tuple(blocks):
        if os.path.exists(p):
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
    return digest.hexdigest()[:16]


def seed_index(name):
    """Stable per-condition seed index derived from the condition NAME.

    Replaces the old positional index so that adding, removing or reordering
    conditions leaves every other condition's random draws untouched. Sharding
    already relied on a position-independent value; a name hash keeps that and
    additionally survives table edits.
    """
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)


def condition_fingerprint(condition, engine_state, run_params):
    """Identity of ONE condition: its simulated content, the engine DR state it
    samples against, and the run-level knobs that change what an episode is."""
    return _sha({"cond": {k: v for k, v in condition.items() if k not in _COSMETIC},
                 "engine": engine_state, "run": run_params})


def code_fingerprint(pkg_dir=None):
    """Hash of the simulation source plus the git description.

    Recorded, never enforced: only a human can say whether a given edit was
    physics-neutral. Stripping render-only meshes provably was (bit-identical
    CSVs); moving the observation anchor to the chest provably was not. So a
    mismatch is surfaced loudly at plan time and carried into the report, and
    the call is left to the reader.
    """
    pkg_dir = pkg_dir or os.path.dirname(os.path.abspath(__file__))
    digest = hashlib.sha1()
    for name in _CODE_FILES:
        path = os.path.join(pkg_dir, name)
        digest.update(open(path, "rb").read() if os.path.exists(path) else b"")
    def git(*args):
        try:
            return subprocess.run(("git",) + args, cwd=pkg_dir, capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"source_sha": digest.hexdigest()[:12],
            "git_commit": git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", *(
                os.path.join(pkg_dir, f) for f in _CODE_FILES)))}


def build_manifest(tables, engine_state, run_params):
    """{table title -> {condition name -> fingerprint}} plus provenance."""
    return {"seeding": "name_hash",
            "protocol": PROTOCOL_VERSION,
            "code": code_fingerprint(),
            "run_params": run_params,
            "engine_state": engine_state,
            "tables": {title: {c["name"]: condition_fingerprint(c, engine_state, run_params)
                               for c in table}
                       for title, table in tables}}


def load_manifest(path):
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None


def save_manifest(path, manifest):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, path)   # atomic: concurrent shards never see a partial file


def classify(new_manifest, old_manifest, title):
    """-> (reusable, changed, new, stale) condition-name lists for one table."""
    want = new_manifest["tables"].get(title, {})
    have = (old_manifest or {}).get("tables", {}).get(title, {})
    reusable = sorted(n for n, fp in want.items() if have.get(n) == fp)
    changed = sorted(n for n, fp in want.items() if n in have and have[n] != fp)
    fresh = sorted(n for n in want if n not in have)
    stale = sorted(n for n in have if n not in want)
    return reusable, changed, fresh, stale


def legacy_seeding(old_manifest, has_rows):
    """True when episodes on disk predate name-hash seeding — either no manifest
    at all (a run recorded before this module existed) or an older scheme.

    Those episodes were drawn from positional streams. That is safe, because a
    condition's episodes are kept or dropped as a whole and never mixed within
    one condition, but the run is no longer reproducible from the table alone,
    so the caller must say so rather than imply the dir is self-describing.
    """
    if not has_rows:
        return False
    return old_manifest is None or old_manifest.get("seeding") != "name_hash"


def describe_drift(old_manifest, new_manifest):
    """Lines describing how the recorded build differs from this one; empty when
    nothing worth reporting changed. Kept separate from `git_dirty`, which says
    only 'unreproducible', not 'different'."""
    if not old_manifest:
        return []
    old, new = old_manifest.get("code", {}), new_manifest["code"]
    lines = []
    for key, label in (("source_sha", "engine/runner/conditions source"),
                       ("git_commit", "git commit")):
        if old.get(key) != new.get(key):
            lines.append(f"{label}: {old.get(key) or '?'} -> {new.get(key) or '?'}")
    if old.get("git_dirty") or new.get("git_dirty"):
        lines.append("working tree was/is dirty — the source hash is the only "
                     "real identity here, the commit alone does not pin it")
    old_proto = old_manifest.get("protocol", 1)
    if old_proto != PROTOCOL_VERSION:
        lines.append(f"PROTOCOL {old_proto} -> {PROTOCOL_VERSION}: the episodes on "
                     f"disk measure a different experiment under the same condition "
                     f"names. Re-run into a FRESH --out-dir; do not top up.")
    return lines
