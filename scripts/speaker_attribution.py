"""Attach diarization speakers to an ASR transcript's words.

Pure Python on purpose: the Parakeet server imports mlx at module level, and
this is the part worth testing in CI, where no MLX runtime exists.

The diarizer answers "who spoke when" as per-frame activity probabilities; the
ASR answers "what was said" as timed words. The two are joined post hoc, by the
mean activity over each word's interval — the same rule as mlx-audio's
`examples/nemotron_diarization_asr.py --mode timestamps`.

Two cases are left for the smoothing pass rather than guessed at here:

- A word in **overlapping speech** is not given the louder speaker. Its
  `candidates` say who was talking, and `overlap` says why it is unassigned.
- A word just **outside the diarizer's speech boundary**. RNN-T timestamps are
  emission times, so the first word of a turn is routinely stamped a frame
  before the diarizer hears the speaker start ("To" / "Hi," / "What's"), and a
  turn's last word a frame after it stops.

`smooth()` resolves those from the words around them, and marks every word it
resolved `speaker_inferred: true` so a consumer can tell evidence from repair.
"""

from __future__ import annotations

import math

SENTENCE_END = (".", "?", "!", "…")
# A pause longer than this starts a new phrase even without punctuation.
PHRASE_GAP_SECONDS = 0.5


def label(index: int) -> str:
    return f"speaker_{index}"


def attribute(words: list[dict], probs: list[list[float]], frame_seconds: float,
              threshold: float = 0.5) -> list[dict]:
    """Return a copy of `words` with `speaker`, `speaker_score`, `overlap`, `candidates`.

    `words` carry `start`/`end` in seconds; `probs` is `(frames, speakers)`.
    """
    if not math.isfinite(frame_seconds) or frame_seconds <= 0:
        raise ValueError("frame_seconds must be positive and finite")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    frames = len(probs)
    timeline_end = frames * frame_seconds
    out = []
    for word in words:
        item = {**word, "speaker": None, "speaker_score": 0.0,
                "overlap": False, "candidates": []}
        start, end = float(word["start"]), float(word["end"])
        # A zero-length word uses the frame holding its emission time.
        stop = end if end > start else start + frame_seconds
        left, right = max(0.0, start), min(timeline_end, stop)
        if right > left:
            first = max(0, math.floor(left / frame_seconds))
            last = min(frames, math.ceil(right / frame_seconds))
            speakers = len(probs[first]) if first < frames else 0
            scores = [0.0] * speakers
            active = set()
            total = 0.0
            for frame in range(first, last):
                f_start = frame * frame_seconds
                weight = min(f_start + frame_seconds, right) - max(f_start, left)
                if weight <= 0:
                    continue
                total += weight
                row = probs[frame]
                on = [s for s in range(speakers) if row[s] > threshold]
                if len(on) > 1:
                    item["overlap"] = True
                active.update(on)
                for s in range(speakers):
                    scores[s] += row[s] * weight
            if total > 0 and speakers:
                scores = [v / total for v in scores]
                best = max(range(speakers), key=scores.__getitem__)
                item["speaker_score"] = round(scores[best], 4)
                item["candidates"] = [label(s) for s in sorted(active)]
                if not item["overlap"] and scores[best] > threshold:
                    item["speaker"] = label(best)
        out.append(item)
    return out


def _starts_phrase(words: list[dict], i: int) -> bool:
    if i == 0:
        return True
    prev = words[i - 1]
    return (str(prev.get("word", "")).rstrip().endswith(SENTENCE_END)
            or float(words[i]["start"]) - float(prev["end"]) > PHRASE_GAP_SECONDS)


def smooth(words: list[dict]) -> list[dict]:
    """Give an unassigned word the speaker of the phrase it belongs to.

    A word that starts a phrase (first word, after sentence punctuation, or
    after a pause) belongs with what follows it; any other word belongs with
    what precedes it. Either way the speaker must be one the diarizer heard
    during that word, when it heard anyone — so crosstalk is never handed to
    a speaker who was not talking.
    """
    words = [dict(w) for w in words]
    for i, word in enumerate(words):
        if word["speaker"] is not None:
            continue
        before = next((words[j]["speaker"] for j in range(i - 1, -1, -1)
                       if words[j]["speaker"]), None)
        after = next((words[j]["speaker"] for j in range(i + 1, len(words))
                      if words[j]["speaker"]), None)
        order = (after, before) if _starts_phrase(words, i) else (before, after)
        allowed = set(word["candidates"])
        for speaker in order:
            if speaker and (not allowed or speaker in allowed):
                word["speaker"] = speaker
                word["speaker_inferred"] = True
                break
    return words


def turns(sentences: list[list[dict]]) -> list[dict]:
    """Split each ASR sentence wherever its speaker changes.

    Sentence boundaries are kept, so a long monologue stays subtitle-sized
    rather than becoming one segment; consecutive segments may share a speaker.
    """
    segments = []
    for sentence in sentences:
        run: list[dict] = []
        for word in sentence:
            if run and word["speaker"] != run[-1]["speaker"]:
                segments.append(_segment(run))
                run = []
            run.append(word)
        if run:
            segments.append(_segment(run))
    for index, segment in enumerate(segments):
        segment["id"] = index
    return segments


def _segment(run: list[dict]) -> dict:
    return {
        "id": 0,
        "start": float(run[0]["start"]),
        "end": float(run[-1]["end"]),
        "text": " ".join(str(w["word"]).strip() for w in run).strip(),
        "speaker": run[0]["speaker"],
        "overlap": any(w["overlap"] for w in run),
        "words": run,
    }


def speaker_summary(segments: list[dict]) -> list[dict]:
    """Per speaker: talk time and first appearance, in order of arrival.

    This is what a later naming step works from — who introduced themselves
    first, who talked most.
    """
    stats: dict[str, dict] = {}
    for segment in segments:
        speaker = segment["speaker"]
        if speaker is None:
            continue
        entry = stats.setdefault(speaker, {"id": speaker, "first_start": segment["start"],
                                           "talk_seconds": 0.0, "segments": 0, "words": 0})
        entry["talk_seconds"] += segment["end"] - segment["start"]
        entry["segments"] += 1
        entry["words"] += len(segment["words"])
    for entry in stats.values():
        entry["talk_seconds"] = round(entry["talk_seconds"], 2)
        entry["first_start"] = round(entry["first_start"], 3)
    return sorted(stats.values(), key=lambda e: e["first_start"])


def render_text(segments: list[dict]) -> str:
    """One paragraph per speaker turn, consecutive same-speaker segments joined.

    An identified speaker is written by name; the rest keep `speaker_N`.
    """
    paragraphs: list[tuple[str | None, list[str]]] = []
    for segment in segments:
        who = segment.get("speaker_name") or segment["speaker"]
        if paragraphs and paragraphs[-1][0] == who:
            paragraphs[-1][1].append(segment["text"])
        else:
            paragraphs.append((who, [segment["text"]]))
    return "\n\n".join(f"{speaker or 'unknown'}: {' '.join(texts)}"
                       for speaker, texts in paragraphs)
