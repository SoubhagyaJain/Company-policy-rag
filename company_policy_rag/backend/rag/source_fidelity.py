"""Source fidelity: keep an answer faithful to what the source defines.

Retrieval ranks passages by similarity to the question. That says nothing about
whether a passage belongs to a list the source defines, whether an explanation
arrived with its heading, or whether a term in the answer exists in the evidence.
This module answers those questions deterministically, from text alone:

* reconstruct an explicit numbered list the source defines, following it across
  consecutive chunks of the same document even when retrieval returned only part;
* keep navigation pages (tables of contents) out of the evidence;
* give a chunk that starts mid-item the heading it continues from;
* check a generated answer for missing, extra or renamed list members, terms that
  appear in neither evidence nor question, citations that do not cover their
  sentence, and numeric disagreements between sources.

Nothing here calls a model.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from backend.models.chunk import Chunk
from backend.models.rag import ScoredChunk

_COUNT_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_COUNT = r"(?P<count>\d{1,2}|" + "|".join(_COUNT_WORDS) + r")"
# "five chunking strategies", "3 steps", "the two main types of leave"
_COUNTED_PHRASE = re.compile(
    rf"\b{_COUNT}\s+(?P<modifiers>(?:[A-Za-z][\w-]*\s+){{0,4}}?)(?P<noun>[A-Za-z][\w-]*s)\b",
    re.IGNORECASE,
)
# "1) Fixed-size chunking", "2. Foo", "#3) Bar", "Step 4: Baz"
_ITEM_LINE = re.compile(
    r"^[ \t]*(?:#|step[ \t]+)?(?P<index>\d{1,2})[ \t]*[.):][ \t]+(?P<label>[^\n]*[A-Za-z][^\n]*?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_LEADER_LINE = re.compile(r"\.{6,}\s*\d{1,4}\s*$")
_SOURCE_TAG = re.compile(
    r"[\[(](?:Visual\s+)?Sources?\s+(\d+(?:\s*(?:,|and|&)\s*(?:(?:Visual\s+)?Sources?\s+)?\d+)*)[^\])\n]{0,160}[\])]",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
_ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5, "sixth": 6, "6th": 6, "seventh": 7, "7th": 7, "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9, "tenth": 10, "10th": 10,
}
_ORDINAL_REFERENCE = re.compile(
    r"\b(?:the\s+)?(?:(?P<ordinal>" + "|".join(_ORDINALS) + r"|last)\s+(?:one|item|point|option|\w+)"
    r"|(?:number|item|point|option|step|#)\s*(?P<number>\d{1,2}|" + "|".join(_COUNT_WORDS) + r"|one))\b",
    re.IGNORECASE,
)
_EXPLAIN_INTENT = re.compile(
    r"\b(?:explain|describe|compare|contrast|differen\w+|how|why|detail\w*|elaborate|walk me through|"
    r"pros|cons|advantages?|disadvantages?|trade-?offs?|when to use|works?)\b",
    re.IGNORECASE,
)
_STOP = frozenset(
    "a an and are as at be been but by can could did do does for from had has have how i if in into is it its "
    "may might more most must no not of on one or our out should so some such than that the their them then "
    "there these they this those to up us use used using was we were what when where which who why will with "
    "would you your also all any each other both between about over under after before during per via".split()
)
# Words that only phrase a request for a list; they add no further condition to it.
_REQUEST_WORDS = frozenset(
    "list name give tell show provide enumerate mention mentioned state stated cover covered discuss discussed "
    "present presented define defined outline outlined exist document documents guide guidebook book pdf file "
    "text source sources section chapter according different various main please briefly short".split()
)
# Words an answer uses to organise itself; never source terminology.
_ANSWER_SCAFFOLD = frozenset(
    "overview summary conclusion note notes example examples step steps answer key point points source sources "
    "visual document section page advantages disadvantages pros cons definition purpose limitations inference "
    "however therefore additionally finally first second third next then thus".split()
)


def normalize(text: str | None) -> str:
    """NFKC text: folds PDF ligatures ("ﬁve" -> "five") that break every literal match."""
    return unicodedata.normalize("NFKC", text or "")


def _stem(word: str) -> str:
    w = word.casefold()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es") and not w.endswith(("ses", "ies")):
        return w[:-2] if w[-3] in "sxz" or w.endswith(("ches", "shes")) else w[:-1]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _content_stems(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(normalize(text)) if len(w) > 2 and w.casefold() not in _STOP}


def _to_count(raw: str) -> int | None:
    raw = raw.casefold()
    return int(raw) if raw.isdigit() else _COUNT_WORDS.get(raw)


def is_navigation_chunk(text: str | None) -> bool:
    """A table of contents or index page: related to every question, evidence for none."""
    lines = [line for line in normalize(text).splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    leaders = sum(1 for line in lines if _LEADER_LINE.search(line))
    return leaders >= 3 and leaders / len(lines) >= 0.4


def drop_navigation_chunks(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """Remove navigation pages, unless nothing else was retrieved."""
    kept = [sc for sc in chunks if not is_navigation_chunk(sc.chunk.text)]
    return kept or chunks


# ── Document order ──────────────────────────────────────────────────────────


class DocumentOrder:
    """Reading order of every chunk in the docstore, per document."""

    def __init__(self, docstore: dict[str, Chunk] | None) -> None:
        self._sequences: dict[str, list[Chunk]] = {}
        self._position: dict[str, tuple[str, int]] = {}
        by_document: dict[str, list[Chunk]] = {}
        for chunk in (docstore or {}).values():
            meta = chunk.metadata
            if getattr(meta, "node_role", None) == "parent":
                continue
            by_document.setdefault(meta.document_id, []).append(chunk)
        for document_id, chunks in by_document.items():
            chunks.sort(key=lambda c: (c.metadata.page_number or 0, c.metadata.chunk_index or 0, c.id))
            self._sequences[document_id] = chunks
            for position, chunk in enumerate(chunks):
                self._position[chunk.id] = (document_id, position)

    def following(self, chunk_id: str, limit: int) -> list[Chunk]:
        located = self._position.get(chunk_id)
        if located is None:
            return []
        document_id, position = located
        return self._sequences[document_id][position + 1 : position + 1 + limit]

    def previous(self, chunk_id: str) -> Chunk | None:
        located = self._position.get(chunk_id)
        if located is None or located[1] == 0:
            return None
        return self._sequences[located[0]][located[1] - 1]


# ── Structure reconstruction ────────────────────────────────────────────────


@dataclass
class StructureItem:
    index: int
    label: str
    chunk_id: str
    excerpt: str = ""
    # Chunk the excerpt was read from; differs from ``chunk_id`` when the heading
    # closes one chunk and its explanation opens the next.
    excerpt_chunk_id: str = ""


@dataclass
class SourceStructure:
    """A numbered list the source defines, in the source's own words and order."""

    noun_phrase: str
    expected: int
    items: list[StructureItem] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    # The source sentence that announces the list, e.g. "Here are five chunking strategies for RAG:".
    introducer: str = ""
    # Stem of the noun the question used for the list when it differs from the source's.
    asked_as: str = ""

    @property
    def complete(self) -> bool:
        return len(self.items) >= self.expected

    @property
    def labels(self) -> list[str]:
        return [item.label for item in self.items]


def _clean_label(raw: str) -> str:
    return normalize(raw).strip().strip("-*:.; ").strip()


def _item_lines(text: str) -> list[tuple[int, str, int, int]]:
    """(index, label, start, end) of every numbered item line in normalised text."""
    return [
        (int(m.group("index")), _clean_label(m.group("label")), m.start(), m.end())
        for m in _ITEM_LINE.finditer(text)
        if _clean_label(m.group("label"))
    ]


def _query_phrase(query: str) -> tuple[int | None, set[str], set[str]]:
    """(requested count, stems of plural nouns, stems of every content word) in the question."""
    text = normalize(query)
    counted = _COUNTED_PHRASE.search(text)
    count = _to_count(counted.group("count")) if counted else None
    nouns = {_stem(w) for w in _WORD.findall(text) if len(w) > 3 and w.casefold().endswith("s") and w.casefold() not in _STOP}
    return count, nouns, _content_stems(text)


def _boilerplate_lines(chunks: Iterable[Chunk]) -> set[str]:
    """Running headers/footers: the same first line on several chunks, and bare page numbers."""
    first_lines: dict[str, int] = {}
    for chunk in chunks:
        lines = [line.strip() for line in normalize(chunk.text).splitlines() if line.strip()]
        if lines:
            first_lines[lines[0]] = first_lines.get(lines[0], 0) + 1
    return {line for line, seen in first_lines.items() if seen >= 2}


def _excerpt(body: str, boilerplate: set[str], max_words: int = 40) -> str:
    """First sentence or two of an item's explanation, taken verbatim from the source."""
    lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and line.strip() not in boilerplate and not line.strip().isdigit()
    ]
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if not text:
        return ""
    picked: list[str] = []
    words = 0
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence_words = len(sentence.split())
        if picked and words + sentence_words > max_words:
            break
        picked.append(sentence)
        words += sentence_words
        if words >= 12:
            break
    excerpt = " ".join(picked)
    return excerpt if excerpt.endswith((".", "!", "?")) else " ".join(excerpt.split()[:max_words])


def find_structure(
    query: str,
    context: list[ScoredChunk],
    order: DocumentOrder,
    *,
    max_extra_chunks: int = 3,
) -> SourceStructure | None:
    """Reconstruct the numbered list the question asks about, if the source defines one.

    The list is anchored where a retrieved chunk introduces it ("Here are five
    chunking strategies:") and is then followed through the next chunks of the
    same document until every promised item is found. A list is only accepted
    when its introducer names what the question asks for, so a numbered list
    that merely happens to be in the evidence is never treated as the answer.
    """
    query_count, query_nouns, query_stems = _query_phrase(query)
    if not query_nouns and query_count is None:
        return None

    for scored in context:
        text = normalize(scored.chunk.text)
        for intro in _COUNTED_PHRASE.finditer(text):
            expected = _to_count(intro.group("count"))
            if not expected or expected < 2:
                continue
            if query_count is not None and query_count != expected:
                continue
            modifiers = _content_stems(intro.group("modifiers"))
            shares_modifier = bool(modifiers & query_stems)
            # "five fine-tuning techniques" must not answer "chunking techniques".
            if modifiers and not shares_modifier:
                continue
            same_noun = _stem(intro.group("noun")) in query_nouns
            # The asker may say "techniques" where the source says "strategies";
            # the same count of the same thing is still that list.
            same_counted_thing = query_count == expected and shares_modifier
            if not (same_noun or same_counted_thing):
                continue
            line_start = text.rfind("\n", 0, intro.start()) + 1
            line_end = text.find("\n", intro.end())
            structure = _collect_items(
                noun_phrase=re.sub(r"\s+", " ", (intro.group("modifiers") + intro.group("noun"))).strip(),
                expected=expected,
                anchor=scored.chunk,
                anchor_offset=intro.end(),
                order=order,
                max_extra_chunks=max_extra_chunks,
            )
            if structure is not None:
                structure.introducer = text[line_start : line_end if line_end != -1 else len(text)].strip()
                asked = _COUNTED_PHRASE.search(normalize(query))
                structure.asked_as = _stem(asked.group("noun")) if asked else ""
                return structure
    return None


def _collect_items(
    *,
    noun_phrase: str,
    expected: int,
    anchor: Chunk,
    anchor_offset: int,
    order: DocumentOrder,
    max_extra_chunks: int,
) -> SourceStructure | None:
    sequence = [anchor, *order.following(anchor.id, expected + max_extra_chunks)]
    boilerplate = _boilerplate_lines(sequence)
    items: list[StructureItem] = []
    bodies: list[str] = []  # where each item's explanation begins
    last_position = 0
    idle = 0

    def add_body(part: str, source: Chunk) -> None:
        # The excerpt is taken from one chunk only, so it can be cited exactly.
        if not items[-1].excerpt_chunk_id and len(_WORD.findall(part)) >= 4:
            items[-1].excerpt_chunk_id = source.id
            bodies[-1] = part

    for position, chunk in enumerate(sequence):
        text = normalize(chunk.text)
        cursor = anchor_offset if position == 0 else 0
        progressed = False
        ended = False
        for index, label, start, end in _item_lines(text):
            if start < cursor:
                continue
            if index == 1 and items:
                # A second list begins here; the first one is over.
                add_body(text[cursor:start], chunk)
                ended = True
                break
            if index != len(items) + 1:
                continue
            if items:
                add_body(text[cursor:start], chunk)
            items.append(StructureItem(index=index, label=label, chunk_id=chunk.id))
            bodies.append("")
            cursor = end
            progressed = True
            last_position = position
            if len(items) == expected:
                break
        if ended:
            break
        if items:
            add_body(text[cursor:], chunk)
        if len(items) == expected:
            break
        idle = 0 if progressed else idle + 1
        # An introducer with no list after it, or a list that stopped early.
        if idle >= 2:
            break

    if not items:
        return None
    used = sequence[: last_position + 1]
    if len(items) == expected and last_position + 1 < len(sequence):
        # The last item's explanation usually runs onto the next chunk.
        tail = sequence[last_position + 1]
        tail_text = normalize(tail.text)
        tail_items = _item_lines(tail_text)
        add_body(tail_text[: tail_items[0][2]] if tail_items else tail_text, tail)
        used = used + [tail]
    for item, body in zip(items, bodies):
        item.excerpt = _excerpt(body, boilerplate)
    return SourceStructure(noun_phrase=noun_phrase, expected=expected, items=items, chunks=used)


def structure_scope(query: str, structure: SourceStructure) -> str:
    """How the question relates to a source-defined list.

    ``list``     asks for the members themselves ("what are the five strategies?")
    ``members``  asks about every member ("explain the five strategies")
    ``related``  asks something else that involves the list ("which strategy uses X?")

    Only the first two are answers that must contain every member.
    """
    asked = _content_stems(query) - _content_stems(structure.introducer) - _content_stems(structure.noun_phrase)
    asked -= {_stem(w) for w in (*_COUNT_WORDS, *_REQUEST_WORDS)} | {structure.asked_as}
    explain_words = {_stem(w) for w in _WORD.findall(" ".join(_EXPLAIN_INTENT.findall(normalize(query))))}
    if asked - explain_words:
        return "related"
    return "members" if _EXPLAIN_INTENT.search(normalize(query)) else "list"


def apply_structure_to_context(
    context: list[ScoredChunk],
    structure: SourceStructure,
    max_chunks: int,
) -> list[ScoredChunk]:
    """Evidence for a source-defined list: its own chunks first, in source order.

    Other retrieved chunks are kept only while room remains, so a passage that is
    merely related to the topic cannot crowd out a member of the list.
    """
    by_id = {sc.chunk.id: sc for sc in context}
    anchor_score = max((sc.score for sc in context), default=1.0)
    ordered = [
        by_id.get(chunk.id) or ScoredChunk(chunk=chunk, score=anchor_score)
        for chunk in structure.chunks
    ]
    structure_ids = {chunk.id for chunk in structure.chunks}
    room = max(0, max_chunks - len(ordered))
    rest = [sc for sc in context if sc.chunk.id not in structure_ids][:room]
    return ordered + rest


def continuation_labels(context: list[ScoredChunk], order: DocumentOrder) -> dict[str, str]:
    """For chunks that open mid-item, the numbered heading they continue from.

    Page-bound chunking puts a heading at the foot of one chunk and its
    explanation at the head of the next, which then reaches the model unnamed.
    """
    labels: dict[str, str] = {}
    for scored in context:
        previous = order.previous(scored.chunk.id)
        if previous is None:
            continue
        own_text = normalize(scored.chunk.text)
        own_items = _item_lines(own_text)
        lead = own_text[: own_items[0][2]] if own_items else own_text
        if len(_WORD.findall(lead)) < 8:
            continue
        previous_text = normalize(previous.text)
        previous_items = _item_lines(previous_text)
        if not previous_items:
            continue
        index, label, _start, end = previous_items[-1]
        if _has_heading_after(previous_text[end:]):
            continue
        labels[scored.chunk.id] = f"{index}) {label}"
    return labels


def _has_heading_after(text: str) -> bool:
    """A short unpunctuated line following a finished sentence: an unnumbered heading."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for previous, line in zip(lines, lines[1:]):
        if (
            previous.endswith((".", "!", "?"))
            and len(line) < 60
            and not line.endswith((".", "!", "?", ",", ";", ":"))
            and not line.isdigit()
            and line[:1].isupper()
            and len(line.split()) >= 2
        ):
            return True
    return False


def format_structure_block(structure: SourceStructure, source_index: dict[str, int]) -> str:
    """The list as the source defines it, for the top of the evidence."""
    found = len(structure.items)
    lines = [
        f'SOURCE-DEFINED LIST: "{structure.noun_phrase}" '
        f"({found} of {structure.expected} found in the evidence, source order)"
    ]
    for item in structure.items:
        tag = f" [Source {source_index[item.chunk_id]}]" if item.chunk_id in source_index else ""
        lines.append(f"{item.index}. {item.label}{tag}")
    if structure.complete:
        lines.append("Use exactly these items, names and order. Do not add, drop, rename, merge or split items.")
    else:
        lines.append(
            f"Only these {found} are in the retrieved evidence. Name them, state that the other "
            f"{structure.expected - found} are not in the retrieved evidence, and do not guess them."
        )
    return "\n".join(lines)


def render_structure_answer(
    structure: SourceStructure,
    source_index: dict[str, int],
    *,
    with_excerpts: bool = False,
) -> str:
    """The list answer written from the source itself: exact names, order and citations."""
    count_word = next((word for word, value in _COUNT_WORDS.items() if value == structure.expected), str(structure.expected))
    if structure.complete:
        lines = [f"The {count_word} {structure.noun_phrase} are:", ""]
    else:
        lines = [
            f"The source lists {count_word} {structure.noun_phrase}, but the retrieved evidence "
            f"contains only {len(structure.items)} of them:",
            "",
        ]
    for item in structure.items:
        if with_excerpts and item.excerpt:
            # Cite where the name is stated and where its explanation was read.
            sources = sorted(
                {source_index[cid] for cid in (item.chunk_id, item.excerpt_chunk_id) if cid in source_index}
            )
            tag = f" [Source {', '.join(map(str, sources))}]" if sources else ""
            lines.append(f"{item.index}. **{item.label}**: {item.excerpt}{tag}")
        else:
            tag = f" [Source {source_index[item.chunk_id]}]" if item.chunk_id in source_index else ""
            lines.append(f"{item.index}. {item.label}{tag}")
    if not structure.complete:
        lines += ["", f"The remaining {structure.expected - len(structure.items)} are not in the retrieved evidence."]
    return "\n".join(lines)


# ── Answer checks ───────────────────────────────────────────────────────────


@dataclass
class FidelityFindings:
    missing_items: list[str] = field(default_factory=list)
    extra_items: list[str] = field(default_factory=list)
    unsupported_terms: list[str] = field(default_factory=list)
    citation_errors: list[str] = field(default_factory=list)
    unreported_conflicts: list[str] = field(default_factory=list)

    @property
    def structure_violated(self) -> bool:
        return bool(self.missing_items or self.extra_items)

    @property
    def clean(self) -> bool:
        return not (
            self.missing_items or self.extra_items or self.unsupported_terms
            or self.citation_errors or self.unreported_conflicts
        )


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize(text).casefold())


_ANSWER_ITEM = re.compile(r"^[ \t]*(?:\d{1,2}[.)]|[-*•])[ \t]+(?P<body>[^\n]+)$", re.MULTILINE)


def _answer_item_labels(answer: str) -> list[str]:
    """The name each list line of the answer leads with."""
    labels = []
    for match in _ANSWER_ITEM.finditer(normalize(answer)):
        body = _SOURCE_TAG.sub("", match.group("body")).strip()
        bold = re.match(r"\*\*(.+?)\*\*", body)
        head = bold.group(1) if bold else re.split(r"\s[-–—]\s|:|\.\s|\(", body, maxsplit=1)[0]
        head = head.strip(" *_`:.-")
        if head:
            labels.append(head)
    return labels


def check_structure_answer(answer: str, structure: SourceStructure) -> tuple[list[str], list[str]]:
    """(source items the answer left out, list members the source does not define)."""
    squashed_answer = _squash(answer)
    missing = [item.label for item in structure.items if _squash(item.label) not in squashed_answer]

    # Words every label shares ("chunking") identify nothing; compare on the rest.
    label_stems = [_content_stems(label) for label in structure.labels]
    shared = set.intersection(*label_stems) if len(label_stems) > 1 else set()
    distinctive = [stems - shared or stems for stems in label_stems]
    extra = []
    for head in _answer_item_labels(answer):
        head_stems = _content_stems(head) - shared
        if not head_stems:
            continue
        if any(_squash(label) in _squash(head) or _squash(head) in _squash(label) for label in structure.labels):
            continue
        if any(len(head_stems & stems) / len(stems) >= 0.6 for stems in distinctive if stems):
            continue
        extra.append(head)
    return missing, extra


_TERM_PATTERNS = (
    re.compile(r"`([^`\n]{2,40})`"),
    re.compile(r"\b[A-Za-z]+(?:_[A-Za-z0-9]+)+\b"),
    re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}\b|\b[a-z]+[A-Z][A-Za-z0-9]*\b"),
    re.compile(r"\b[A-Z][A-Z0-9]{2,9}\b"),
    # Hyphenated names with a capital or digit after the hyphen: "D-Wave", "GPT-4".
    re.compile(r"\b[A-Z][A-Za-z0-9]*(?:-(?:[A-Z]|\d)[A-Za-z0-9]*)+\b"),
)
# A capitalised name mid-sentence: preceded by a lowercase word or a comma.
_MID_SENTENCE_NAME = re.compile(r"(?<=[a-z,] )([A-Z][a-z]+(?:[ -][A-Z][a-z]+){0,3})\b")


def find_unsupported_terms(answer: str, evidence: str, question: str = "", limit: int = 8) -> list[str]:
    """Names and technical terms in the answer that appear in neither evidence nor question."""
    clean = _SOURCE_TAG.sub(" ", normalize(answer))
    clean = re.sub(r"^\s*#+\s.*$", " ", clean, flags=re.MULTILINE)
    haystack = _squash(evidence + " " + question)
    candidates: list[str] = []
    for pattern in _TERM_PATTERNS:
        candidates += [m.group(1) if m.groups() else m.group(0) for m in pattern.finditer(clean)]
    candidates += [m.group(1) for m in _MID_SENTENCE_NAME.finditer(clean.replace("**", ""))]

    unsupported: list[str] = []
    seen: set[str] = set()
    for term in candidates:
        key = _squash(term)
        if len(key) < 3 or key in seen or term.casefold() in _ANSWER_SCAFFOLD:
            continue
        seen.add(key)
        if key in haystack or _squash(_stem(term)) in haystack:
            continue
        # A multi-word name is supported when each of its words is.
        words = [w for w in _WORD.findall(term) if w.casefold() not in _STOP]
        if len(words) > 1 and all(_squash(w) in haystack or _squash(_stem(w)) in haystack for w in words):
            continue
        unsupported.append(term.strip())
    # "Retrieval-Augmented" inside "Retrieval-Augmented Generation" is one finding.
    distinct = [
        term
        for term in unsupported
        if not any(term != other and _squash(term) in _squash(other) for other in unsupported)
    ]
    return distinct[:limit]


def find_citation_errors(
    answer: str,
    context: list[ScoredChunk],
    *,
    min_coverage: float = 0.5,
    limit: int = 5,
) -> list[str]:
    """Sentences whose cited source does not contain what the sentence says."""
    errors: list[str] = []
    stems_by_source = {index: _content_stems(sc.chunk.text) for index, sc in enumerate(context, start=1)}
    for sentence in re.split(r"(?<=[.!?\]])\s+(?=[A-Z0-9*-])|\n+", normalize(answer)):
        cited = {int(n) for tag in _SOURCE_TAG.finditer(sentence) for n in re.findall(r"\d+", tag.group(1))}
        cited = {n for n in cited if n in stems_by_source}
        if not cited:
            continue
        # "The evidence does not mention X" describes the evidence; no source states it.
        if _ABOUT_EVIDENCE.search(sentence):
            continue
        claim = _content_stems(_SOURCE_TAG.sub(" ", sentence))
        if len(claim) < 4:
            continue
        evidence = set().union(*(stems_by_source[n] for n in cited))
        coverage = len(claim & evidence) / len(claim)
        if coverage < min_coverage:
            snippet = re.sub(r"\s+", " ", _SOURCE_TAG.sub("", sentence)).strip()[:110]
            tags = ", ".join(f"Source {n}" for n in sorted(cited))
            errors.append(f"{tags} does not state: \"{snippet}\"")
            if len(errors) >= limit:
                break
    return errors


_ABOUT_EVIDENCE = re.compile(
    r"\b(?:evidence|sources?|documents?|handbook|excerpts?|context|passages?|text)\b[^.]{0,80}"
    r"\b(?:not|no|cannot|can't|unable|lacks?|without|silent)\b"
    r"|\b(?:not|no|cannot|can't|unable)\b[^.]{0,80}\b(?:evidence|sources?|documents?|handbook|excerpts?|context)\b",
    re.IGNORECASE,
)

_QUANTITY = re.compile(
    r"(?P<value>\$\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?)\s*(?P<unit>%|percent|days?|weeks?|months?|years?|hours?|minutes?|"
    r"dollars?|usd|gb|mb|tokens?|times?)?",
    re.IGNORECASE,
)


def find_numeric_conflicts(context: list[ScoredChunk], *, min_overlap: float = 0.6) -> list[tuple[int, str, int, str, set[str]]]:
    """Pairs of sources that state different quantities for the same sentence.

    Returns (source A, value A, source B, value B, shared wording).
    """
    statements: list[tuple[int, str, set[str], str]] = []
    for index, scored in enumerate(context, start=1):
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", normalize(scored.chunk.text)):
            quantities = [
                (m.group("value").replace(" ", "").replace(",", ""), (m.group("unit") or "").casefold().rstrip("s"))
                for m in _QUANTITY.finditer(sentence)
                if m.group("unit") or m.group("value").startswith("$")
            ]
            if len(quantities) != 1:
                continue
            value, unit = quantities[0]
            words = _content_stems(_QUANTITY.sub(" ", sentence))
            if len(words) >= 3:
                statements.append((index, f"{value} {unit}".strip(), words, unit))
    conflicts = []
    for position, (source_a, value_a, words_a, unit_a) in enumerate(statements):
        for source_b, value_b, words_b, unit_b in statements[position + 1 :]:
            if source_a == source_b or unit_a != unit_b or value_a == value_b:
                continue
            shared = words_a & words_b
            if len(shared) / min(len(words_a), len(words_b)) >= min_overlap:
                conflicts.append((source_a, value_a, source_b, value_b, shared))
    return conflicts


def format_conflict_block(conflicts: list[tuple[int, str, int, str, set[str]]]) -> str:
    if not conflicts:
        return ""
    lines = ["SOURCES DISAGREE (report every value with its source; do not choose one):"]
    lines += [f"- [Source {a}] says {va}; [Source {b}] says {vb}" for a, va, b, vb, _shared in conflicts[:3]]
    return "\n".join(lines)


def check_answer(
    answer: str,
    context: list[ScoredChunk],
    question: str = "",
    structure: SourceStructure | None = None,
    *,
    enforce_members: bool = True,
    extra_evidence: str = "",
) -> FidelityFindings:
    """Every deterministic fidelity check for one generated answer.

    ``enforce_members`` is False when the question only involves a source-defined
    list without asking for all of it: an invented member is still an error, a
    member left unmentioned is not.
    """
    findings = FidelityFindings()
    if not answer or not answer.strip():
        return findings
    evidence = "\n".join(sc.chunk.text for sc in context) + "\n" + extra_evidence
    if structure is not None:
        missing, findings.extra_items = check_structure_answer(answer, structure)
        findings.missing_items = missing if enforce_members else []
        evidence += "\n" + "\n".join(structure.labels)
    findings.unsupported_terms = find_unsupported_terms(answer, evidence, question)
    findings.citation_errors = find_citation_errors(answer, context)
    squashed = _squash(answer)
    for source_a, value_a, source_b, value_b, _shared in find_numeric_conflicts(context):
        stated_a, stated_b = _squash(value_a) in squashed, _squash(value_b) in squashed
        if stated_a != stated_b:
            findings.unreported_conflicts.append(
                f"Source {source_a} says {value_a} but Source {source_b} says {value_b}; the answer gives only one."
            )
    return findings


# ── Conversation ────────────────────────────────────────────────────────────


def resolve_ordinal_reference(query: str, previous_answer: str | None) -> str | None:
    """Rewrite "explain the second one" using the numbered list of the previous answer.

    Returns the question with the reference replaced by that item's name, or None
    when the question has no ordinal reference or the previous answer has no list.
    """
    if not previous_answer:
        return None
    text = normalize(query)
    match = _ORDINAL_REFERENCE.search(text)
    if not match:
        return None
    items = {index: label for index, label, _s, _e in _item_lines(normalize(previous_answer))}
    if len(items) < 2:
        return None
    if match.group("ordinal"):
        ordinal = match.group("ordinal").casefold()
        position = max(items) if ordinal == "last" else _ORDINALS[ordinal]
    else:
        raw = match.group("number").casefold()
        position = 1 if raw == "one" else (_to_count(raw) or 0)
    label = items.get(position)
    if not label:
        return None
    name = _SOURCE_TAG.sub("", label)
    name = re.sub(r"\*\*(.+?)\*\*.*", r"\1", name)
    name = re.split(r"\s[-–—]\s|:", name, maxsplit=1)[0].strip(" *_`.")
    if not name:
        return None
    return (text[: match.start()] + name + text[match.end() :]).strip()
