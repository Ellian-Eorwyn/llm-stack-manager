#!/usr/bin/env python3
"""Cleaning, sectioning, chunking, and the four dataset shapes.

The cleaning battery is ported from `~/ft-data/build_dataset.py`, the script
that actually produced the 177-row academic-voice corpus. Every regex here
earned its place against a real export, and the comments say which one -- that
provenance is the point, because the artefacts differ by converter rather than
by subject and a cleaner written from first principles misses all of them.

What changed in the port: this emits `{"messages": [...]}` directly rather than
a rendered ChatML string. The original rendered to `text` and a second script
parsed it back to `messages`, and that round-trip was pure loss surface -- eight
rows died in it. Rendering belongs at train time, where the template's arguments
are known.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# The cleaning battery
# ---------------------------------------------------------------------------

FRONTMATTER = re.compile(r"^\s*---\n.*?\n---\n", re.S)
HEADING = re.compile(r"^(#{1,6})\s+(\S.*?)\s*$")
#: docx export artefact: a bare `####` is an empty paragraph marker.
HEADING_ONLY = re.compile(r"^(#{1,6})\s*$")
MD_ESCAPE = re.compile(r"\\([\"'\-\[\]|])")
FOOTNOTE_REF = re.compile(r"\[\^\d+\]")
FOOTNOTE_DEF = re.compile(r"^\[\^\d+\]:.*?(?=^\[\^\d+\]:|\Z)", re.S | re.M)
WIKILINK = re.compile(r"\[\[([^[^\]]+?)\]\]")
MDLINK = re.compile(r"\[([^\]]*)\]\(https?://[^)]*\)")
#: The residue a markdown export leaves where a note anchor used to be:
#: `persons.1(#endnote-1)`. 181 of them across 44 files in the first corpus.
NOTE_ANCHOR = re.compile(r"\d+\(#(?:end|foot)note-\d+\)")
IMAGE_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")
BOLD_LINE = re.compile(r"^\*\*([^*]+)\*\*\s*$")
INLINE_BOLD = re.compile(r"\*\*([^*]+)\*\*")
CARET_FOOTNOTE = re.compile(r"\^\d+\^")
#: `^th^`, `^st^` — ordinals, kept as text rather than deleted.
CARET_SUP = re.compile(r"\^(\w{1,3})\^")
BACKLINK = re.compile(r"\[↑\]\(#[^)]*\)")
#: `———. Same-author bibliography continuation`
EMDASH_REF = re.compile(r"^——+\s*\.?\s")
HRULE = re.compile(r"^\s*-{3,}\s*$")
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
#: Numbered bibliography entries: number, space, Capital, then a year within
#: about 40 characters. Deliberately narrow so prose lists ("1. First, I
#: argue...") survive.
NUM_REF = re.compile(
    r"^\s*\d+\.\s+[A-Z][\w&'\-]*(?:\s+[\w&'\-]+){0,4}\s[.,:;](?:\s.*?\d{4})?\b")
LISTY_LINE = re.compile(r"^(>\s*|-\s|\*\s|\d+\.\s|\d+\)\s)")
#: Publisher front matter that PDF-to-markdown converters repeat per page.
BOILERPLATE_LINE = re.compile(
    r"^(to cite this article|to link to this article|published online|"
    r"submit your article|article views|view crossmark|cite this article|"
    r"recommended citations|share this article|access options|how to cite|"
    r"doi:\s*\d|source pdf:|ocr model:|mode:|render dpi:)", re.I)
#: One or two short Capitalized-Name words: an author byline, not a heading.
#: A capitalised word, then one or two more that are either a full word or a
#: middle initial — "Jane Smith", "Jane Q. Smith", "Mary Jane Smith". The
#: original could not express the initial and let those lines through as prose.
BYLINE = re.compile(
    r"^[A-Z][\w'\-]+(?:\s+(?:[A-Z]\.?|[A-Z][\w'\-]+)){1,2}\s*\.?\s*$")
#: ...except that a Title Case heading of two or three words matches the same
#: shape. "The Repair Shop" is a section, "Ellian Eorwyn" is a byline, and the
#: regex alone reads both as bylines and drops the heading. Names do not begin
#: with an article or a preposition; headings routinely do.
NOT_A_NAME_FIRST_WORD = frozenset("""
a an the this that these those on in of at by for from with without within
against beyond after before during toward towards between among about into
why how what when where who which some other another no not all
notes introduction conclusion abstract discussion methods results background
""".split())


def _is_byline(text: str) -> bool:
    if not BYLINE.match(text):
        return False
    return text.split()[0].lower() not in NOT_A_NAME_FIRST_WORD
AFFILIATION_LINE = re.compile(
    r"^(department|school|faculty|college|institute|lab|laboratory)\s+of\b", re.I)
PSEUDO_HEADING = re.compile(
    r"^(references|notes|works cited|bibliography|works\s+referenced)\s*\.?\s*$", re.I)
#: A speaker turn produced by `ingest._captions_to_text` or present in a
#: transcript already: `Ellie: text`.
SPEAKER_TURN = re.compile(r"^([A-Z][\w .'-]{0,40}):\s+(?=\S)")

#: A line recurring this many times is a running header, not content.
RECURRING_LINE_THRESHOLD = 3
#: Only short lines are considered for that: a repeated paragraph is plagiarism,
#: a repeated title is pagination.
RECURRING_LINE_MAX_WORDS = 25


def clean_line(line: str) -> tuple[str | None, bool]:
    """Clean one line. Returns (text or None to drop, is_pseudo_heading)."""
    s = line.strip()
    if not s:
        return None, False
    bold = BOLD_LINE.match(s)
    if bold:
        inner = bold.group(1)
        if _is_byline(inner) or AFFILIATION_LINE.match(inner):
            return None, False
        return None, True  # `**Title**` as a heading, from docx conversion
    if HRULE.match(s) or TABLE_ROW.match(s):
        return None, False
    if EMDASH_REF.match(s) or BOILERPLATE_LINE.match(s):
        return None, False
    if len(s.split()) <= 4 and _is_byline(s):
        return None, False
    if BACKLINK.search(s):
        return None, False

    text = re.sub(r"^#{1,6}[ \t]*", "", s)
    text = WIKILINK.sub(r"\1", text)
    text = MDLINK.sub(r"\1", text)
    text = IMAGE_MD.sub("", text)
    text = MD_ESCAPE.sub(r"\1", text)
    text = NOTE_ANCHOR.sub("", text)
    text = FOOTNOTE_REF.sub("", text)
    text = CARET_FOOTNOTE.sub("", text)
    text = CARET_SUP.sub(r"\1", text)
    text = INLINE_BOLD.sub(r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return (text or None), False


def parse_sections(md: str) -> list[tuple[str, list[str]]]:
    """(heading, lines) in document order, cleaned, with reference sections cut."""
    md = FRONTMATTER.sub("", md)
    md = FOOTNOTE_DEF.sub("", md)

    counts = Counter(
        line.strip() for line in md.split("\n")
        if line.strip() and len(line.split()) <= RECURRING_LINE_MAX_WORDS)
    recurring = {line for line, n in counts.items() if n >= RECURRING_LINE_THRESHOLD}

    sections: list[tuple[str, list[str]]] = []
    current: list[str] = []
    heading: str | None = None

    def flush():
        nonlocal current, heading
        sections.append((heading or "", current))
        current, heading = [], None

    for raw in md.split("\n"):
        stripped = raw.strip()
        if stripped in recurring:
            continue
        # docx artefact: `####\tProse` is a paragraph wearing a heading marker.
        raw = re.sub(r"^#{1,6}[ \t]*\t", "", raw)
        if HEADING_ONLY.match(raw.strip()):
            continue
        text, is_pseudo = clean_line(raw)
        if is_pseudo:
            flush()
            bold = BOLD_LINE.match(raw.strip())
            heading = MD_ESCAPE.sub(r"\1", bold.group(1) if bold else raw.strip()).lower()
            continue
        match = HEADING.match(raw.strip())
        if match:
            flush()
            heading = MD_ESCAPE.sub(r"\1", match.group(2)).strip().lower()
            continue
        if text is not None and PSEUDO_HEADING.match(text):
            flush()
            heading = text.lower()
            continue
        if text is not None:
            current.append(text)
    flush()

    kept = []
    for head, lines in sections:
        if any(word in head for word in
               ("reference", "note", "works cited", "bibliography")):
            continue
        nonblank = [l for l in lines if l]
        # An unnamed trailing section that is half reference lines is a
        # bibliography whose heading the converter ate.
        if nonblank and sum(1 for l in nonblank if NUM_REF.match(l)) / len(nonblank) >= 0.5:
            continue
        kept.append((head, lines))
    return kept


def prose_ratio(lines: list[str]) -> float:
    """Fraction of non-blank lines that are not list, quote or reference lines."""
    nonblank = [l for l in lines if l.strip()]
    if not nonblank:
        return 0.0
    prose = sum(1 for l in nonblank
                if not LISTY_LINE.match(l.strip()) and not NUM_REF.match(l.strip()))
    return prose / len(nonblank)


def paragraphs(lines: list[str]) -> list[str]:
    """Join hard-wrapped lines into blank-line-separated paragraphs."""
    paras, buf = [], []
    for line in lines:
        if line.strip():
            buf.append(line.strip())
        elif buf:
            paras.append(" ".join(buf))
            buf = []
    if buf:
        paras.append(" ".join(buf))
    return paras


def chunk_paragraphs(pairs, min_words: int, max_words: int):
    """Greedy-pack (paragraph, heading) into [min_words, max_words] chunks.

    A paragraph longer than the cap is split on sentence boundaries rather than
    truncated. Truncation is what produced the four rows that ended mid-clause
    in the first corpus, each of which taught the model to trail off.
    """
    chunks, buf, words, head = [], [], 0, None

    def flush():
        nonlocal buf, words, head
        if buf:
            chunks.append((" ".join(buf), head or ""))
        buf, words, head = [], 0, None

    for para, para_head in pairs:
        count = len(para.split())
        if count > max_words:
            if buf and words + count > max_words:
                flush()
            for sentence in re.split(r"(?<=[.!?])\s+", para):
                sentence_words = len(sentence.split())
                if buf and words + sentence_words > max_words:
                    flush()
                if not buf:
                    head = para_head
                buf.append(sentence)
                words += sentence_words
            flush()
            continue
        if buf and words + count > max_words:
            flush()
        if not buf:
            head = para_head
        buf.append(para)
        words += count
    flush()

    long_enough = [c for c in chunks if len(c[0].split()) >= min_words]
    return long_enough or chunks


def normalize_sig(text: str) -> str:
    """Near-duplicate fingerprint: alphanumerics only, first 400 + last 400.

    Exact matching misses re-submitted drafts that differ by a few words or a
    footnote marker. This catches those while keeping genuinely distinct
    sections that merely open alike -- two sections sharing a 600-character
    prefix scored 0.54 on their full text and are correctly kept apart.
    """
    # Strip note markers *before* reducing to alphanumerics: `[^3]` would
    # otherwise reduce to a bare `3` and survive, which is exactly the
    # difference between two submissions of the same draft.
    text = NOTE_ANCHOR.sub("", text)
    text = FOOTNOTE_REF.sub("", text)
    text = CARET_FOOTNOTE.sub("", text)
    core = re.sub(r"[^a-z0-9]", "", text.lower())
    return core[:400] + "|" + core[-400:]


# ---------------------------------------------------------------------------
# instruction synthesis
# ---------------------------------------------------------------------------

TASK_FRAMINGS = [
    "Write a section on “{section}”. Develop the argument fully across several paragraphs.",
    "Draft a section titled “{section}”. Maintain a consistent voice and build each claim carefully.",
    "In your own voice, write the section “{section}”. Be specific and careful with hedging.",
    "Produce the full section “{section}”. Use sustained paragraphs and clear transitions between claims.",
    "Compose a section on “{section}”.",
]
#: Long headings read like sentences and do not fit the slot grammars above.
NEUTRAL_FRAMING = ('Write a sustained section titled "{section}". '
                   "Develop the argument across several paragraphs.")
#: Used when a chunk has no heading at all and no model is available to caption it.
UNTITLED_FRAMING = "Continue the argument in your own voice, in sustained paragraphs."

#: Above this many characters, a heading is a sentence, not a title.
LONG_TITLE_CHARS = 45


def instruction_for(section: str, rng: random.Random) -> tuple[str, str]:
    """(instruction, method) for a heading. Method is 'heading' or 'untitled'.

    The slot repairs are not cosmetic. A heading beginning "the" produces
    "Write the the repair shop"; a heading carrying quote marks makes the
    quoted slot unreadable. Both appeared in the first corpus.
    """
    if not section.strip():
        return UNTITLED_FRAMING, "untitled"
    slot = re.sub(r"^the\s+", "", section.strip())
    slot = slot.replace('"', "").replace("\u201c", "").replace("\u201d", "").strip()
    if not slot:
        return UNTITLED_FRAMING, "untitled"
    framing = NEUTRAL_FRAMING if len(section) > LONG_TITLE_CHARS else rng.choice(TASK_FRAMINGS)
    return framing.format(section=slot), "heading"


#: Grammar accidents the slot repairs are meant to prevent. `validate` greps for
#: these, because a malformed instruction is invisible in a row count.
GRAMMAR_SMELLS = (" the the ", " a a ", " an an ", " a introduction",
                  " section section ", "“”", '""')


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------

@dataclass
class BuildResult:
    rows: list[dict] = field(default_factory=list)
    dropped: Counter = field(default_factory=Counter)
    methods: Counter = field(default_factory=Counter)
    per_source: Counter = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)


def _chunks_for(text: str, min_words: int, max_words: int,
                min_prose_ratio: float, dropped: Counter):
    """Sections to (chunk, heading), dropping the sections that are not prose."""
    pairs = []
    for heading, lines in parse_sections(text):
        if prose_ratio(lines) < min_prose_ratio:
            dropped["section is mostly lists or references"] += 1
            continue
        for para in paragraphs(lines):
            pairs.append((para, heading))
    return chunk_paragraphs(pairs, min_words, max_words)


def build_style(documents, system: str, min_words: int, max_words: int,
                min_prose_ratio: float, seed: int, caption=None) -> BuildResult:
    """Prose in, instruction synthesised, assistant turn is the prose verbatim."""
    result = BuildResult()
    rng = random.Random(seed)
    seen: set[str] = set()
    for doc in documents:
        for chunk, heading in _chunks_for(doc["text"], min_words, max_words,
                                          min_prose_ratio, result.dropped):
            signature = normalize_sig(chunk)
            if signature in seen:
                result.dropped["near-duplicate of an earlier chunk"] += 1
                continue
            seen.add(signature)
            instruction, method = instruction_for(heading, rng)
            if method == "untitled" and caption is not None:
                written = caption(chunk)
                if written:
                    instruction, method = written, "llm"
            result.methods[method] += 1
            result.per_source[doc["name"]] += 1
            result.rows.append({
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": chunk},
                ],
                "meta": {"source": doc["name"], "section": heading,
                         "instruction_method": method},
            })
    return result


def build_dialogue(documents, system: str, assistant_speaker: str,
                   max_words: int, seed: int) -> BuildResult:
    """A transcript into user/assistant turns.

    Refuses an unattributed transcript rather than guessing. Which side of a
    conversation the model is being taught to be is the entire content of the
    dataset; inferring it from turn order would be a coin flip applied silently.
    """
    result = BuildResult()
    target = assistant_speaker.strip().lower()
    for doc in documents:
        turns = _speaker_turns(doc["text"])
        if not turns:
            result.dropped["no speaker labels in the transcript"] += 1
            result.notes.append(
                f"{doc['name']}: no `Speaker: text` turns found — a transcript "
                f"without attribution cannot be split into roles.")
            continue
        speakers = {who.lower() for who, _ in turns}
        if target not in speakers:
            result.dropped[f"no turns by {assistant_speaker!r}"] += 1
            result.notes.append(
                f"{doc['name']}: --assistant-speaker {assistant_speaker!r} never "
                f"speaks. Present: {', '.join(sorted(speakers))}.")
            continue
        for index, (who, said) in enumerate(turns):
            if who.lower() != target or index == 0:
                continue
            prompt_who, prompt = turns[index - 1]
            if prompt_who.lower() == target:
                continue
            said = " ".join(said.split()[:max_words])
            result.per_source[doc["name"]] += 1
            result.methods["transcript"] += 1
            result.rows.append({
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": said},
                ],
                "meta": {"source": doc["name"], "speaker": who,
                         "instruction_method": "transcript"},
            })
    return result


def _speaker_turns(text: str) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for block in re.split(r"\n\s*\n", text):
        block = " ".join(block.split())
        match = SPEAKER_TURN.match(block)
        if not match:
            if turns:
                turns[-1] = (turns[-1][0], f"{turns[-1][1]} {block}".strip())
            continue
        turns.append((match.group(1).strip(), block[match.end():].strip()))
    return [(who, said) for who, said in turns if said]


def build_qa(documents, system: str, min_words: int, max_words: int,
             min_prose_ratio: float, seed: int, ask) -> BuildResult:
    """Questions written against each chunk, answers grounded in that chunk.

    `ask` is required: unlike `style`, there is no deterministic fallback. A
    question invented from a heading would not be answerable from the chunk,
    which is the only thing that makes this shape worth building.
    """
    result = BuildResult()
    seen: set[str] = set()
    for doc in documents:
        for chunk, heading in _chunks_for(doc["text"], min_words, max_words,
                                          min_prose_ratio, result.dropped):
            signature = normalize_sig(chunk)
            if signature in seen:
                result.dropped["near-duplicate of an earlier chunk"] += 1
                continue
            seen.add(signature)
            question = ask(chunk, heading)
            if not question:
                result.dropped["no question could be generated"] += 1
                continue
            result.methods["llm"] += 1
            result.per_source[doc["name"]] += 1
            result.rows.append({
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": chunk},
                ],
                "meta": {"source": doc["name"], "section": heading,
                         "instruction_method": "llm"},
            })
    return result


def build_raw(documents, min_words: int, max_words: int,
              min_prose_ratio: float) -> BuildResult:
    """Plain text chunks. No instructions, for continued pretraining."""
    result = BuildResult()
    seen: set[str] = set()
    for doc in documents:
        for chunk, _ in _chunks_for(doc["text"], min_words, max_words,
                                    min_prose_ratio, result.dropped):
            signature = normalize_sig(chunk)
            if signature in seen:
                result.dropped["near-duplicate of an earlier chunk"] += 1
                continue
            seen.add(signature)
            result.methods["raw"] += 1
            result.per_source[doc["name"]] += 1
            result.rows.append({"text": chunk, "meta": {"source": doc["name"]}})
    return result
