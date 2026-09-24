"""The voice-profile store under models/voiceprints/ (git-ignored).

    samples.npz    every enrolled clip: embedding, label, recording, start, and
                   the diarized cluster it came from ("" for clips taken from
                   hand-labelled transcripts).
    config.json    decisions that outlive a rebuild: which labels are one
                   person (`same_person`) and which are left out (`exclude`).
    profiles.json  what the transcription server reads; always rebuilt from
                   the two above, never edited by hand.

Writes are atomic (temp file + rename): the server re-reads profiles.json
whenever its mtime changes and must never see half a file.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

import speaker_identity

STACK = Path(__file__).resolve().parents[1]
# VOICEPRINT_STORE points everything at another store — a test copy, say.
DEFAULT_DIR = Path(os.environ.get("VOICEPRINT_STORE") or STACK / "models" / "voiceprints")
FIELDS = ("vectors", "names", "sessions", "starts", "clusters")


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=path.suffix)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_samples(store: Path = DEFAULT_DIR) -> dict:
    path = store / "samples.npz"
    if not path.exists():
        return {"vectors": np.zeros((0, 256), np.float32), "names": np.array([], str),
                "sessions": np.array([], str), "starts": np.array([], float),
                "clusters": np.array([], str)}
    data = dict(np.load(path))
    if "clusters" not in data:  # stores written before diarized enrollment existed
        data["clusters"] = np.array([""] * len(data["names"]))
    return {k: data[k] for k in FIELDS}


def save_samples(samples: dict, store: Path = DEFAULT_DIR) -> None:
    def write(tmp):
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **{k: samples[k] for k in FIELDS})
    _atomic(store / "samples.npz", write)


def replace_cluster(samples: dict, session: str, cluster: str, name: str,
                    starts: list[float], vectors: list) -> dict:
    """Drop whatever `cluster` of `session` was enrolled as, and enroll it as `name`.

    Naming a speaker twice (a correction) must not leave the first name's
    clips behind in the other person's profile.
    """
    keep = ~((samples["sessions"] == session) & (samples["clusters"] == cluster))
    out = {k: samples[k][keep] for k in FIELDS}
    if vectors:
        n = len(vectors)
        out["vectors"] = np.concatenate([out["vectors"], np.asarray(vectors, np.float32)])
        out["names"] = np.concatenate([out["names"], np.array([name] * n)])
        out["sessions"] = np.concatenate([out["sessions"], np.array([session] * n)])
        out["starts"] = np.concatenate([out["starts"], np.asarray(starts, float)])
        out["clusters"] = np.concatenate([out["clusters"], np.array([cluster] * n)])
    return out


def load_config(store: Path = DEFAULT_DIR) -> dict:
    path = store / "config.json"
    config = json.loads(path.read_text()) if path.exists() else {}
    if not config:
        # Before config.json existed, the decisions were only recorded in the
        # profiles they produced.
        old = store / "profiles.json"
        if old.exists():
            data = json.loads(old.read_text())
            config = {"same_person": data.get("same_person", []), "exclude": data.get("excluded", [])}
    return {"same_person": config.get("same_person", []), "exclude": config.get("exclude", [])}


def save_config(config: dict, store: Path = DEFAULT_DIR) -> None:
    _atomic(store / "config.json",
            lambda tmp: Path(tmp).write_text(json.dumps(config, indent=1) + "\n"))


def canonical(label: str, config: dict) -> str:
    """The profile a label is enrolled under, after `same_person`."""
    return speaker_identity.parse_same_person(config["same_person"]).get(label, label)


def rebuild(store: Path = DEFAULT_DIR, evaluate: bool = True, model: str = "",
            source: str = "") -> dict:
    """profiles.json from samples.npz and config.json. Returns the result written."""
    samples = load_samples(store)
    config = load_config(store)
    excluded = set(config["exclude"])
    mask = np.array([n not in excluded for n in samples["names"]], bool)
    names = samples["names"][mask].tolist()
    sessions = samples["sessions"][mask].tolist()
    vectors = samples["vectors"][mask]
    aliases = speaker_identity.parse_same_person(config["same_person"])
    people = [aliases.get(n, n) for n in names]

    profiles, report = speaker_identity.build_profiles(people, sessions, vectors.tolist())
    for person, variants in speaker_identity.variants_of(aliases).items():
        if person in profiles:
            profiles[person]["labels"] = variants
    result = {"model": model, "source": source, "same_person": config["same_person"],
              "excluded": config["exclude"], "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "profiles": profiles, "enrollment": report,
              "similar_pairs": speaker_identity.similar_pairs(profiles)}
    if evaluate and len(set(sessions)) > 1:
        result["evaluation"] = speaker_identity.evaluate(people, sessions, vectors.tolist())
        result["threshold"] = result["evaluation"]["threshold"]
        result["margin"] = result["evaluation"]["margin"]
    _atomic(store / "profiles.json",
            lambda tmp: Path(tmp).write_text(json.dumps(result, indent=1)))
    return result
