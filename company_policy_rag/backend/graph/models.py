"""Node/edge models, stable ID construction, and traversal rules for the policy graph.

Node IDs are namespaced and deterministic so a rebuild of the same corpus produces
byte-identical identifiers. That is what makes graph builds idempotent: re-ingesting
a document cannot silently duplicate structure.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.utils.hashing import compute_string_hash

# Length of the hashed component in a SECTION node id. Section paths can be long
# and contain arbitrary punctuation, so they are hashed rather than embedded.
_SECTION_HASH_LEN = 12

_WHITESPACE_RE = re.compile(r"\s+")
# Trailing separators only. Closing brackets are deliberately excluded: a
# subclause id like "9.1(b)" ends in a paren that belongs to the identifier.
# Interior punctuation is significant too \u2014 "HR-104" and "hr 104" must not
# collapse into the same entity.
_TRAILING_PUNCT_RE = re.compile(r"[\s.,;:!?\u2013\u2014-]+$")
_LEADING_PUNCT_RE = re.compile(r"^[\s\u00a7]+")
# Bracket pairs stripped only when they wrap the entire value, so "(FLSA)"
# normalises to "flsa" while "9.1(b)" keeps its subclause marker.
_BRACKET_PAIRS = (("(", ")"), ("[", "]"), ("{", "}"))


class NodeType(str, Enum):
    DOCUMENT = "document"
    POLICY = "policy"
    SECTION = "section"
    CLAUSE = "clause"
    CHUNK = "chunk"
    ENTITY = "entity"
    TOPIC = "topic"


class EdgeType(str, Enum):
    CONTAINS = "contains"
    PARENT_OF = "parent_of"
    NEXT = "next"
    REFERENCES = "references"
    MENTIONS = "mentions"
    TAGGED = "tagged"
    GOVERNED_BY = "governed_by"


Provenance = Literal["structural", "reference", "llm"]


class EdgeSpec(BaseModel):
    """Traversal rules for one edge type.

    ``base_weight`` multiplies into path confidence; ``max_neighbors`` caps fan-out
    per node so a single dense edge type cannot consume the traversal budget.
    """

    base_weight: float
    max_neighbors: int
    traverse_reverse: bool = True
    # True when the edge reaches a shared hub node (entity/topic/policy) that is
    # only useful as a two-hop bridge between chunks, never as a destination.
    is_bridge: bool = False


# Traversal rules. REFERENCES is weighted highest because it is an author-asserted
# link; MENTIONS/TAGGED are weakest because shared vocabulary is weak evidence and
# is further damped by IDF at build time.
EDGE_SPECS: dict[EdgeType, EdgeSpec] = {
    EdgeType.CONTAINS: EdgeSpec(base_weight=0.50, max_neighbors=6),
    EdgeType.PARENT_OF: EdgeSpec(base_weight=0.90, max_neighbors=4),
    EdgeType.NEXT: EdgeSpec(base_weight=0.80, max_neighbors=2),
    EdgeType.REFERENCES: EdgeSpec(base_weight=1.00, max_neighbors=5),
    EdgeType.MENTIONS: EdgeSpec(base_weight=0.40, max_neighbors=3, is_bridge=True),
    EdgeType.TAGGED: EdgeSpec(base_weight=0.30, max_neighbors=3, is_bridge=True),
    EdgeType.GOVERNED_BY: EdgeSpec(base_weight=0.35, max_neighbors=3, is_bridge=True),
}

# Edge types carrying thematic (rather than structural) signal. Community
# detection runs on these alone: CONTAINS/NEXT are dense and complete within every
# document, so including them collapses every community onto a document boundary.
SEMANTIC_EDGE_TYPES: frozenset[EdgeType] = frozenset(
    {EdgeType.REFERENCES, EdgeType.MENTIONS, EdgeType.GOVERNED_BY}
)


def normalize_label(value: str) -> str:
    """Casefold, collapse whitespace, and strip enclosing punctuation.

    Used for every globally-scoped node id (entity, topic, policy) so that
    ``"FLSA"``, ``"flsa "`` and ``"(FLSA)"`` resolve to one node.
    """
    if not value:
        return ""
    collapsed = _WHITESPACE_RE.sub(" ", value).strip()

    # Peel bracket pairs that wrap the whole value, e.g. "(FLSA)" -> "FLSA".
    peeled = True
    while peeled and len(collapsed) > 1:
        peeled = False
        for opener, closer in _BRACKET_PAIRS:
            if collapsed.startswith(opener) and collapsed.endswith(closer):
                collapsed = collapsed[1:-1].strip()
                peeled = True
                break

    collapsed = _LEADING_PUNCT_RE.sub("", collapsed)
    collapsed = _TRAILING_PUNCT_RE.sub("", collapsed)
    return collapsed.casefold()


def document_node_id(document_id: str) -> str:
    return f"doc:{document_id}"


def policy_node_id(policy_id: str) -> str:
    return f"policy:{normalize_label(policy_id)}"


def section_node_id(document_id: str, section_path: str) -> str:
    return f"sec:{document_id}:{compute_string_hash(normalize_label(section_path))[:_SECTION_HASH_LEN]}"


def clause_node_id(document_id: str, clause_id: str) -> str:
    return f"clause:{document_id}:{normalize_label(clause_id)}"


def chunk_node_id(chunk_id: str) -> str:
    return f"chunk:{chunk_id}"


def entity_node_id(name: str) -> str:
    return f"entity:{normalize_label(name)}"


def topic_node_id(tag: str) -> str:
    return f"topic:{normalize_label(tag)}"


class GraphNode(BaseModel):
    id: str
    node_type: NodeType
    label: str = ""
    # Chunks this node resolves to. A CHUNK node holds exactly one; SECTION and
    # CLAUSE nodes hold every chunk beneath them; ENTITY/TOPIC nodes hold every
    # chunk mentioning them. Traversal results map back to Chunk objects through
    # this field, which is why the graph never needs to store text.
    chunk_ids: list[str] = Field(default_factory=list)
    document_id: str | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    confidence: float = 1.0
    provenance: Provenance = "structural"
    attrs: dict[str, Any] = Field(default_factory=dict)


class PendingReference(BaseModel):
    """A cross-reference that could not be resolved to exactly one target yet.

    Kept on the graph and replayed after every ingest: a reference to a document
    uploaded later must still resolve, or cross-reference edges would silently
    depend on upload order.
    """

    from_chunk_id: str
    raw_text: str
    target_key: str
    scope_document_id: str | None = None
    scope_policy_id: str | None = None
