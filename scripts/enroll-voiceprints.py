#!/usr/bin/env python3
"""Build voice profiles from transcripts someone has already labelled.

    python scripts/enroll-voiceprints.py macwhisper [--db PATH] [--out DIR]
        [--same-person "Dana=Dana (Acme)"] [--evaluate]
    python scripts/enroll-voiceprints.py rebuild [--same-person ...] [--evaluate]

`rebuild` regroups the clips already in samples.npz — after deciding two
labels are one person, say — without decoding any audio. --same-person and
--exclude are saved in config.json, so later runs need not repeat them.

Reads MacWhisper's library read-only: every transcript line assigned to a
named speaker is a sample of that person's voice, at a known time in a known
file. Generic labels ("Speaker 3") and MacWhisper's per-meeting "Microphone"
stubs are skipped, because they name different people in different
recordings.

Output, under models/voiceprints/ (git-ignored — these are biometric data
about other people and never leave this machine):

- profiles.json  one centroid per person, how many clips and recordings it
                 rests on, and the threshold `--evaluate` chose.
- samples.npz    every accepted clip's embedding, so a profile can be rebuilt
                 or audited without decoding audio again.
- config.json    the --same-person and --exclude decisions.

See voiceprint_store.py for the layout.

Runs in the MLX runtime venv. See docs/voiceprints.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import voiceprint_store  # noqa: E402
from speaker_embedding import SAMPLE_RATE, SpeakerEmbedder, decode  # noqa: E402

STACK = Path(__file__).resolve().parents[1]
MACWHISPER = Path.home() / "Library/Application Support/MacWhisper/Database"
GENERIC = re.compile(r"^speaker\s*\d+$", re.I)

MIN_CLIP = 2.0     # seconds; shorter clips give noisy embeddings
MAX_CLIP = 12.0    # a longer line is cut to this
TRIM = 0.15        # shaved off each end, away from the neighbouring turn
PER_RECORDING = 40  # clips per person per recording, longest first


def macwhisper_clips(db: Path):
    """Yield (session_hex, [audio paths], [(name, start_s, end_s), ...]) per recording."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    media = db.parent / "ExternalMedia"
    lines = con.execute("""
        select hex(t.sessionId), trim(s.name), s.isStub, t.start, t.end
        from transcriptline t join speaker s on s.id = t.speakerID
        join session se on se.id = t.sessionId
        where se.dateDeleted is null
        order by t.sessionId, t.start""").fetchall()
    by_session = defaultdict(list)
    for session, name, stub, start, end in lines:
        by_session[session].append((name, bool(stub), start / 1000.0, end / 1000.0))

    for session, rows in by_session.items():
        labelled = [(n, a, b) for n, stub, a, b in rows if not stub and not GENERIC.match(n)]
        if not labelled:
            continue
        files = dict(con.execute(
            "select type, filename from mediafile where sessionID = x'%s'" % session).fetchall())
        tracks = [f for t, f in con.execute(
            "select type, filename from mediafile where sessionID = x'%s' and type = 'multitrackItem'"
            % session).fetchall()]
        # A meeting has a merged file plus its mic and app tracks on the same
        # timeline; each clip is taken from whichever is loudest there.
        paths = [media / f for f in tracks] or \
                [media / files[k] for k in ("mergedMultitrack", "original", "processed") if k in files][:1]
        paths = [p for p in paths if p.exists()]
        if not paths:
            continue
        # Any other line overlapping a clip — named or not — makes it impure.
        spans = [(a, b, n) for n, _, a, b in rows]
        clips = []
        for name, start, end in labelled:
            a, b = start + TRIM, end - TRIM
            if b - a < MIN_CLIP:
                continue
            if any(n != name and s < b and e > a for s, e, n in spans):
                continue
            clips.append((name, a, min(b, a + MAX_CLIP)))
        if clips:
            yield session, paths, clips


def best_track(tracks: list[np.ndarray], a: int, b: int) -> np.ndarray:
    return max((t[a:b] for t in tracks), key=lambda x: float(np.mean(x * x)) if x.size else -1)


def enroll(args):
    embedder = SpeakerEmbedder(str(args.model))
    samples = []  # (name, session, start, embedding)
    started = time.time()
    counts = defaultdict(lambda: defaultdict(int))
    for index, (session, paths, clips) in enumerate(macwhisper_clips(args.db), 1):
        clips.sort(key=lambda c: c[2] - c[1], reverse=True)
        kept = []
        for name, a, b in clips:
            if counts[name][session] >= PER_RECORDING:
                continue
            counts[name][session] += 1
            kept.append((name, a, b))
        try:
            tracks = [decode(p) for p in paths]
        except subprocess.CalledProcessError as exc:
            print(f"  skip {session[:8]}: cannot decode ({exc.stderr.decode()[:80].strip()})")
            continue
        for name, a, b in kept:
            clip = best_track(tracks, int(a * SAMPLE_RATE), int(b * SAMPLE_RATE))
            if clip.size < MIN_CLIP * SAMPLE_RATE or float(np.sqrt(np.mean(clip * clip))) < 1e-3:
                continue  # past the end of the file, or silence
            samples.append((name, session, round(a, 2), embedder.embed(clip)))
        print(f"  [{index}] {session[:8]} {len(kept)} clips, {time.time() - started:.0f}s", flush=True)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=["macwhisper", "rebuild"])
    parser.add_argument("--db", type=Path, default=MACWHISPER / "main.sqlite")
    parser.add_argument("--model", type=Path, default=STACK / "models/mlx/wespeaker-voxceleb-resnet34-LM")
    parser.add_argument("--out", type=Path, default=voiceprint_store.DEFAULT_DIR)
    parser.add_argument("--same-person", action="append", default=None, metavar="NAME=VARIANT[,VARIANT]",
                        help="labels that are one voice, e.g. 'Dana=Dana (Acme)'. Saved in config.json; "
                             "given at all, it replaces the saved list")
    parser.add_argument("--exclude", action="append", default=None, metavar="LABEL",
                        help="leave a label out, e.g. one whose lines are known to be mislabelled. "
                             "Saved like --same-person")
    parser.add_argument("--evaluate", action="store_true",
                        help="leave-one-recording-out accuracy, and pick the match threshold from it")
    args = parser.parse_args()

    config = voiceprint_store.load_config(args.out)
    if args.same_person is not None:
        config["same_person"] = args.same_person
    if args.exclude is not None:
        config["exclude"] = args.exclude
    voiceprint_store.save_config(config, args.out)

    if args.source == "macwhisper":
        fresh = enroll(args)
        stored = voiceprint_store.load_samples(args.out)
        # Replace only what came from labelled transcripts; speakers named in
        # diarized recordings (a non-empty cluster) are kept.
        keep = stored["clusters"] != ""
        samples = {k: stored[k][keep] for k in voiceprint_store.FIELDS}
        if fresh:
            samples = {
                "vectors": np.concatenate([samples["vectors"], np.stack([f[3] for f in fresh])]),
                "names": np.concatenate([samples["names"], np.array([f[0] for f in fresh])]),
                "sessions": np.concatenate([samples["sessions"], np.array([f[1] for f in fresh])]),
                "starts": np.concatenate([samples["starts"], np.array([f[2] for f in fresh], float)]),
                "clusters": np.concatenate([samples["clusters"], np.array([""] * len(fresh))]),
            }
        voiceprint_store.save_samples(samples, args.out)

    result = voiceprint_store.rebuild(args.out, evaluate=args.evaluate, model=args.model.name,
                                      source=args.source)
    for pair in result["similar_pairs"]:
        print(f"  check: {pair['a']!r} and {pair['b']!r} sound alike ({pair['similarity']})")
    summary = {"profiles": len(result["profiles"]),
               "clips": sum(p["clips"] for p in result["profiles"].values()),
               "same_person": config["same_person"], "excluded": config["exclude"]}
    if "evaluation" in result:
        summary["evaluation"] = {k: v for k, v in result["evaluation"].items() if k != "confusions"}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
