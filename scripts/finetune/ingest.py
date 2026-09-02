#!/usr/bin/env python3
"""Turn a folder of source documents into normalised text, with provenance.

This stage normalises *format*. `shapes.py` normalises *content*. Keeping them
apart matters because the artefacts differ by converter, not by subject: a
docx-to-markdown export leaves bare `####` markers and `^1^` footnotes, a
PDF-to-markdown export leaves running headers and publisher boilerplate, and an
Obsidian note leaves wikilinks. Guessing which cleaner a file needs is the
failure this split removes -- the manifest records which converter produced each
document, so the content stage never has to guess.

Nothing is dropped silently. A file this cannot read is written to the manifest
with a reason, because a corpus that quietly lost a third of its sources looks
exactly like a corpus that was always that size.
"""

from __future__ import annotations

import hashlib
import html.parser
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from common import have, run_command

#: Read directly.
TEXT_SUFFIXES = {".md", ".markdown", ".mdown", ".txt", ".text"}
#: Timed transcripts.
CAPTION_SUFFIXES = {".vtt", ".srt"}
#: Handed to pandoc.
PANDOC_SUFFIXES = {".docx", ".odt", ".rtf", ".epub", ".org", ".rst", ".tex"}
HTML_SUFFIXES = {".html", ".htm"}
PDF_SUFFIXES = {".pdf"}

SUPPORTED = (TEXT_SUFFIXES | CAPTION_SUFFIXES | PANDOC_SUFFIXES
             | HTML_SUFFIXES | PDF_SUFFIXES)

#: Below this, a PDF almost certainly has no text layer -- it is page images.
#: Measured against the character count, not the page count, because a
#: born-digital paper runs to thousands of characters a page and a scan to none.
SCANNED_PDF_CHARS_PER_PAGE = 80


@dataclass
class Document:
    """One source file, normalised."""

    source: Path
    text: str = ""
    converter: str = ""
    sha256: str = ""
    warnings: list[str] = field(default_factory=list)
    skipped: str = ""

    @property
    def ok(self) -> bool:
        return not self.skipped and bool(self.text.strip())

    def record(self) -> dict:
        return {
            "source": str(self.source),
            "name": self.source.name,
            "converter": self.converter,
            "sha256": self.sha256,
            "chars": len(self.text),
            "words": len(self.text.split()),
            "warnings": self.warnings,
            "skipped": self.skipped,
        }


# ---------------------------------------------------------------------------
# converters
# ---------------------------------------------------------------------------

def convert(path: Path) -> Document:
    """Normalise one file. Never raises for a bad input -- it reports."""
    doc = Document(source=path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        doc.skipped = f"unreadable: {exc}"
        return doc
    doc.sha256 = hashlib.sha256(raw).hexdigest()
    suffix = path.suffix.lower()

    try:
        if suffix in TEXT_SUFFIXES:
            doc.text = raw.decode("utf-8", errors="replace")
            doc.converter = "read"
        elif suffix in CAPTION_SUFFIXES:
            doc.text = _captions_to_text(raw.decode("utf-8", errors="replace"), suffix)
            doc.converter = f"captions:{suffix.lstrip('.')}"
        elif suffix in PDF_SUFFIXES:
            _convert_pdf(path, doc)
        elif suffix in HTML_SUFFIXES:
            _convert_html(path, raw, doc)
        elif suffix in PANDOC_SUFFIXES:
            _convert_pandoc(path, doc)
        else:
            doc.skipped = f"unsupported extension {suffix or '(none)'}"
    except Exception as exc:  # a converter crash must not end the walk
        doc.skipped = f"{type(exc).__name__}: {exc}"

    if not doc.skipped and not doc.text.strip():
        doc.skipped = "converted to empty text"
    return doc


def _convert_pdf(path: Path, doc: Document) -> None:
    """PyMuPDF when the interpreter has it, else pdftotext.

    PyMuPDF is preferred only because it can count pages, which is what makes
    the scanned-PDF check meaningful. Both produce plain text; neither will
    recover a page image.
    """
    pages = None
    try:
        import pymupdf  # noqa: PLC0415

        with pymupdf.open(path) as pdf:
            pages = pdf.page_count
            doc.text = "\n\n".join(page.get_text() for page in pdf)
        doc.converter = "pymupdf"
    except ImportError:
        if not have("pdftotext"):
            doc.skipped = ("no PDF converter: install poppler-utils for "
                           "pdftotext, or pymupdf into the training venv")
            return
        code, out, err = run_command(["pdftotext", "-layout", str(path), "-"], timeout=300)
        if code != 0:
            doc.skipped = f"pdftotext failed: {err.strip()[:200]}"
            return
        doc.text = out
        doc.converter = "pdftotext"

    if pages and len(doc.text.strip()) < pages * SCANNED_PDF_CHARS_PER_PAGE:
        doc.skipped = (
            f"only {len(doc.text.strip())} characters across {pages} pages — "
            f"this looks scanned. The stack's GLM-OCR service can read it: "
            f"POST /glmocr/parse on port 5002.")


def _convert_pandoc(path: Path, doc: Document) -> None:
    if not have("pandoc"):
        doc.skipped = f"pandoc is not installed, so {path.suffix} cannot be read"
        return
    code, out, err = run_command(
        ["pandoc", "--from", _pandoc_format(path), "--to", "markdown-raw_html",
         "--wrap=none", str(path)], timeout=300)
    if code != 0:
        doc.skipped = f"pandoc failed: {err.strip()[:200]}"
        return
    doc.text = out
    doc.converter = "pandoc"


def _pandoc_format(path: Path) -> str:
    return {".docx": "docx", ".odt": "odt", ".rtf": "rtf", ".epub": "epub",
            ".org": "org", ".rst": "rst", ".tex": "latex",
            ".html": "html", ".htm": "html"}.get(path.suffix.lower(), "markdown")


def _convert_html(path: Path, raw: bytes, doc: Document) -> None:
    if have("pandoc"):
        _convert_pandoc(path, doc)
        return
    doc.text = _strip_html(raw.decode("utf-8", errors="replace"))
    doc.converter = "html.parser"


class _TextExtractor(html.parser.HTMLParser):
    """Enough of an HTML reader to salvage prose when pandoc is absent."""

    SKIP = {"script", "style", "head", "nav", "footer"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._depth += 1
        elif tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if not self._depth:
            self.parts.append(data)


def _strip_html(text: str) -> str:
    parser = _TextExtractor()
    parser.feed(text)
    return re.sub(r"\n{3,}", "\n\n", "".join(parser.parts))


# ---------------------------------------------------------------------------
# captions
# ---------------------------------------------------------------------------

#: `00:01:02.500 --> 00:01:05.000`, with or without hours, comma or dot.
_TIMECODE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?[.,]\d{1,3}\s*-->")
#: A WebVTT voice span, `<v Ellie>text</v>`, and a plain `Speaker: text` prefix.
_VOICE_SPAN = re.compile(r"<v\s+([^>]+?)>(.*?)(?:</v>)?$", re.S)
_SPEAKER_PREFIX = re.compile(r"^([A-Z][\w .'-]{0,40}):\s+(?=\S)")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def _captions_to_text(text: str, suffix: str) -> str:
    """Timed captions to speaker-attributed prose.

    Timecodes and cue numbers go; speaker labels stay, because `dialogue`
    cannot attribute a turn without them. Consecutive cues from one speaker are
    joined, since a caption break is a display artefact and not a turn.
    """
    lines: list[tuple[str | None, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == "WEBVTT" or _TIMECODE.match(line):
            continue
        # A caption line that is only a number is a cue identifier in both
        # formats -- SRT always numbers them, WebVTT often does.
        if line.isdigit():
            continue
        if line.startswith(("NOTE ", "STYLE", "REGION")):
            continue
        speaker = None
        voice = _VOICE_SPAN.match(line)
        if voice:
            speaker, line = voice.group(1).strip(), voice.group(2).strip()
        else:
            prefix = _SPEAKER_PREFIX.match(line)
            if prefix:
                speaker, line = prefix.group(1).strip(), line[prefix.end():]
        line = _TAG.sub("", line).strip()
        if not line:
            continue
        if lines and speaker in (None, lines[-1][0]):
            lines[-1] = (lines[-1][0], f"{lines[-1][1]} {line}")
        else:
            lines.append((speaker, line))

    return "\n\n".join(f"{who}: {what}" if who else what for who, what in lines)


# ---------------------------------------------------------------------------
# walking
# ---------------------------------------------------------------------------

def walk(source: Path, recursive: bool = True) -> list[Path]:
    """Candidate files, sorted, skipping dotfiles and version control."""
    if source.is_file():
        return [source]
    paths = source.rglob("*") if recursive else source.glob("*")
    return sorted(
        p for p in paths
        if p.is_file()
        and not p.name.startswith(".")
        and not any(part.startswith(".") for part in p.relative_to(source).parts)
    )


def describe_converters() -> list[str]:
    """What this host can currently read. Reported by `ingest`, so a missing
    converter is visible before it turns into a skipped file."""
    lines = [f"  .md .txt          read directly",
             f"  .vtt .srt         built in"]
    try:
        import pymupdf  # noqa: PLC0415, F401
        lines.append("  .pdf              pymupdf")
    except ImportError:
        lines.append("  .pdf              pdftotext" if have("pdftotext")
                     else "  .pdf              UNAVAILABLE: no pymupdf, no pdftotext")
    if have("pandoc"):
        lines.append("  .docx .odt .rtf .epub .html  pandoc")
    else:
        lines.append("  .docx .odt .rtf .epub  UNAVAILABLE: pandoc is not installed")
        lines.append("  .html             html.parser (degraded)")
    return lines
