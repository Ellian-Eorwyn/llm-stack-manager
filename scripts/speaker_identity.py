"""Put names on diarized speakers by comparing voices with enrolled profiles.

A profile is the unit-length mean of a person's clip embeddings
(speaker_embedding.py). A diarized speaker in a new recording is summarised
the same way, and the two are compared by cosine similarity.

Matching is one-to-one and conservative. A name is given only when the score
clears the threshold chosen by leave-one-recording-out evaluation, and beats
the next-best person by a margin — two colleagues with similar voices should
produce "unknown", not a coin toss. No person is given to two speakers.

The runtime half (`match`, `choose_label`, profile loading) is pure Python so
CI can test it without numpy or MLX; building and evaluating use numpy.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

DEFAULT_THRESHOLD = 0.5
DEFAULT_MARGIN = 0.08
_ORG = re.compile(r"^(?P<base>.*?)\s*\((?P<org>[^()]+)\)\s*$")


# --- runtime ---------------------------------------------------------------

def _unit(vec):
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def load(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    for profile in data["profiles"].values():
        profile["centroid"] = _unit(profile["centroid"])
    return data


def match(clusters: dict, profiles: dict, threshold: float = DEFAULT_THRESHOLD,
          margin: float = DEFAULT_MARGIN) -> dict:
    """{cluster_id: vector} -> {cluster_id: {"name", "score", "runner_up", "runner_up_score"}}.

    Every cluster gets an entry; `name` is None where no profile qualifies.
    """
    scored = {}
    for cid, vec in clusters.items():
        vec = _unit(vec)
        scored[cid] = sorted(((_dot(vec, p["centroid"]), name) for name, p in profiles.items()),
                             reverse=True)
    out = {cid: {"name": None, "score": round(s[0][0], 4) if s else 0.0,
                 "runner_up": s[1][1] if len(s) > 1 else None,
                 "runner_up_score": round(s[1][0], 4) if len(s) > 1 else 0.0}
           for cid, s in scored.items()}
    # Strongest pairs first, so a clear match is never displaced by a weak one.
    pairs = sorted(((score, cid, name) for cid, s in scored.items() for score, name in s),
                   reverse=True)
    taken_c, taken_p = set(), set()
    for score, cid, name in pairs:
        if score < threshold:
            break
        if cid in taken_c or name in taken_p:
            continue
        others = [sc for sc, n in scored[cid] if n != name]
        if others and score - others[0] < margin:
            continue
        taken_c.add(cid)
        taken_p.add(name)
        out[cid].update(name=name, score=round(score, 4))
    return out


def clean_spans(diarization: list[dict], limit: int = 20, longest: float = 10.0,
                shortest: float = 1.0) -> dict:
    """Each diarized speaker's stretches with nobody else talking, longest first.

    `diarization` is [{"start", "end", "speaker"}]. A stretch that someone else
    overlaps keeps its larger clean side; a voice sample with two people in it
    matches neither of them.
    """
    spans: dict[str, list] = {}
    for seg in diarization:
        a, b = seg["start"] + 0.1, seg["end"] - 0.1
        for other in diarization:
            if other["speaker"] != seg["speaker"] and other["start"] < b and other["end"] > a:
                left, right = (a, other["start"]), (other["end"], b)
                a, b = max((left, right), key=lambda r: r[1] - r[0])
        if b - a >= shortest:
            spans.setdefault(seg["speaker"], []).append((a, min(b, a + longest)))
    return {sp: sorted(c, key=lambda r: r[0] - r[1])[:limit] for sp, c in spans.items()}


def choose_label(person: str, profile: dict, identified: list[str]) -> str:
    """A voice with context-specific names takes the one its company implies.

    'Dana' carries labels {"Acme": "Dana (Acme)"}; in a recording
    where someone else identified is '… (Acme)', that is the name used.
    """
    labels = profile.get("labels") or {}
    orgs = {m.group("org") for other in identified if other != person
            for m in [_ORG.match(other)] if m}
    for org, label in labels.items():
        if org in orgs:
            return label
    return person


# --- enrollment ------------------------------------------------------------

def parse_same_person(specs: list[str]) -> dict:
    """['Dana=Dana (Acme)'] -> {'Dana': 'Dana', 'Dana (Acme)': 'Dana'}."""
    aliases = {}
    for spec in specs:
        person, _, variants = spec.partition("=")
        person = person.strip()
        aliases[person] = person
        for variant in filter(None, (v.strip() for v in variants.split(","))):
            aliases[variant] = person
    return aliases


def variants_of(aliases: dict) -> dict:
    """{'Dana': {'Acme': 'Dana (Acme)'}} from parse_same_person's output."""
    out: dict[str, dict] = {}
    for variant, person in aliases.items():
        m = _ORG.match(variant)
        if variant != person and m:
            out.setdefault(person, {})[m.group("org")] = variant
    return out


def robust_centroid(vectors):
    """Mean direction after dropping clips that disagree with the majority.

    Catches mislabelled lines: a clip more than three robust deviations below
    the median similarity (or under 0.25 outright) is someone else's voice,
    crosstalk, or noise. Returns (centroid, kept mask, similarities).
    """
    import numpy as np

    x = np.asarray(vectors, dtype=np.float32)
    keep = np.ones(len(x), bool)
    for _ in range(3):
        c = x[keep].mean(0)
        c /= np.linalg.norm(c) or 1.0
        sims = x @ c
        med = float(np.median(sims[keep]))
        mad = float(np.median(np.abs(sims[keep] - med))) * 1.4826
        new = sims >= max(0.25, med - 3 * mad)
        if new.sum() < max(1, len(x) // 2) or (new == keep).all():
            break
        keep = new
    c = x[keep].mean(0)
    return c / (np.linalg.norm(c) or 1.0), keep, x @ c


def build_profiles(people, sessions, vectors):
    import numpy as np

    x = np.asarray(vectors, dtype=np.float32)
    people, sessions = np.asarray(people), np.asarray(sessions)
    profiles, report = {}, {}
    for person in sorted(set(people.tolist())):
        idx = np.flatnonzero(people == person)
        centroid, keep, sims = robust_centroid(x[idx])
        kept = idx[keep]
        recordings = sorted(set(sessions[kept].tolist()))
        profiles[person] = {
            "centroid": [round(float(v), 5) for v in centroid],
            "clips": int(keep.sum()),
            "recordings": len(recordings),
            "consistency": round(float(sims[keep].mean()), 3),
        }
        report[person] = {"clips_offered": int(len(idx)), "clips_rejected": int((~keep).sum())}
    return profiles, report


def similar_pairs(profiles: dict, above: float = 0.6) -> list[dict]:
    """Profile pairs that sound alike — one person under two labels, or a
    recording whose labels are mixed up. Unrelated people rarely pass ~0.55."""
    names = sorted(profiles)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            score = _dot(_unit(profiles[a]["centroid"]), _unit(profiles[b]["centroid"]))
            if score > above:
                out.append({"a": a, "b": b, "similarity": round(score, 3)})
    return sorted(out, key=lambda r: -r["similarity"])


def evaluate(people, sessions, vectors, query_clips=(None, 3), max_wrong=0.02):
    """Leave one recording out, then tune threshold and margin on the result.

    For each recording, every profile is rebuilt without it and each person in
    it is looked up among everyone else. People with clips in two or more
    recordings test recognition. People seen once test strangers: they have no
    profile left, so any name they get is wrong. `query_clips` also tests
    someone who says little (3 clips).

    Each lookup is then replayed through `match`'s rule over a grid of
    thresholds and margins. The pair chosen names the most people correctly
    while wrong names (a known person misnamed, or a stranger named at all)
    stay under `max_wrong` of lookups. Being left unnamed is not wrong: the
    transcript keeps `speaker_N`, which a person can fix.
    """
    import numpy as np

    x = np.asarray(vectors, dtype=np.float32)
    people, sessions = np.asarray(people), np.asarray(sessions)
    names = sorted(set(people.tolist()))
    rng = np.random.default_rng(0)
    # Cleaning uses each person's whole set; slightly optimistic, but the
    # alternative is rebuilding every profile once per recording.
    keep = np.zeros(len(x), bool)
    for name in names:
        idx = np.flatnonzero(people == name)
        keep[idx] = robust_centroid(x[idx])[1]
    sums = {(p, s): x[(people == p) & (sessions == s) & keep].sum(0)
            for p, s in set(zip(people.tolist(), sessions.tolist()))}
    totals = {n: sum((v for (p, _), v in sums.items() if p == n), np.zeros(x.shape[1], np.float32))
              for n in names}

    lookups = {str(q): [] for q in query_clips}   # (known, top_is_right, top, second)
    confusions = []
    for session in sorted(set(sessions.tolist())):
        profiles = {}
        for n in names:
            vec = totals[n] - sums.get((n, session), 0)
            if np.linalg.norm(vec) > 1e-6:
                profiles[n] = vec / np.linalg.norm(vec)
        if len(profiles) < 2:
            continue
        pnames = list(profiles)
        mat = np.stack([profiles[n] for n in pnames])
        for person in sorted(set(people[sessions == session].tolist())):
            idx = np.flatnonzero((people == person) & (sessions == session) & keep)
            if not idx.size:
                continue
            for q in query_clips:
                pick = idx if q is None or len(idx) <= q else rng.choice(idx, q, replace=False)
                c = x[pick].mean(0)
                c /= np.linalg.norm(c) or 1.0
                scores = mat @ c
                order = np.argsort(-scores)
                top = pnames[order[0]]
                known = person in profiles
                lookups[str(q)].append((known, known and top == person,
                                        float(scores[order[0]]), float(scores[order[1]])))
                if q is None and known and top != person:
                    confusions.append({"recording": session[:8], "person": person, "chosen": top,
                                       "score": round(float(scores[order[0]]), 3),
                                       "own": round(float(scores[pnames.index(person)]), 3)})

    def outcome(rows, threshold, margin):
        named = [(k, ok) for k, ok, top, second in rows if top >= threshold and top - second >= margin]
        right = sum(ok for _, ok in named)
        return right, len(named) - right

    # Tuned on every query size at once: a speaker who says three sentences
    # must not be misnamed any more often than one who talks all meeting.
    full = [row for rows in lookups.values() for row in rows]
    # Most right names first, then fewest wrong. Many settings usually tie on
    # both; the middle of that range is kept rather than either edge, since
    # the lowest threshold is nearest to naming a stranger and the highest
    # to leaving a quiet speaker unnamed in a recording not seen here.
    grid = []
    for threshold in np.arange(0.40, 0.96, 0.01):
        for margin in np.arange(0.0, 0.31, 0.01):
            right, wrong = outcome(full, threshold, margin)
            if wrong <= max_wrong * len(full):
                grid.append((right, -wrong, round(float(threshold), 2), round(float(margin), 2)))
    if grid:
        top = max(g[:2] for g in grid)
        tied = [g for g in grid if g[:2] == top]
        threshold = float(np.median(sorted({g[2] for g in tied})))
        threshold = min({g[2] for g in tied}, key=lambda t: abs(t - threshold))
        margins = sorted(g[3] for g in tied if g[2] == threshold)
        margin = margins[len(margins) // 2]
    else:
        threshold, margin = DEFAULT_THRESHOLD, DEFAULT_MARGIN

    summary = {}
    for q, rows in lookups.items():
        known = [r for r in rows if r[0]]
        right, wrong = outcome(rows, threshold, margin)
        summary["all clips" if q == "None" else f"{q} clips"] = {
            "lookups": len(rows), "known": len(known), "strangers": len(rows) - len(known),
            "top1_accuracy": round(sum(r[1] for r in known) / max(1, len(known)), 3),
            "named_right": round(right / max(1, len(known)), 3),
            "named_wrong": round(wrong / max(1, len(rows)), 3),
        }
    return {"threshold": threshold, "margin": margin, "by_query_size": summary,
            "confusions": confusions}
