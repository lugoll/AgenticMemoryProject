"""Shared LLM triple-extraction assets for the unified Neo4j ingest.

Predicate whitelist, extraction prompt, JSON schema, and response parsing are
defined once here and used by the unified ingestion (src/memory/base.py) so
every retrieval view operates on an identically extracted graph.
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

def _normalize_predicate(raw: str, allowed: frozenset[str]) -> str:
    """Map a raw LLM-generated predicate to the nearest allowed predicate.

    Strategy (in order of preference):
    1. Exact match — return as-is.
    2. Canonical aliases — common LLM paraphrases mapped to the intended predicate.
    3. Token-overlap — split both strings on '_' and count shared tokens; take the
       allowed predicate with the highest overlap (ties broken by string length).
    4. Fallback — return 'related_to' if no overlap is found.

    This runs post-LLM so invented free-text predicates ("mastered_recordings_for",
    "was a case of", "executive produced") are silently normalised rather than
    polluting the graph with a long tail of hapax predicates.
    """
    p = raw.strip().lower().replace(" ", "_")

    # 1. Exact match
    if p in allowed:
        return p

    # 2. Hard-coded aliases for common LLM paraphrases
    _ALIASES: dict[str, str] = {
        "works_at":          "educated_at",
        "worked_at":         "educated_at",
        "studied_at":        "educated_at",
        "studies_at":        "educated_at",
        "works_for":         "works_as",
        "worked_as":         "works_as",
        "occupation":        "works_as",
        "profession":        "works_as",
        "is_an":             "is_a",
        "was_a":             "is_a",
        "is_also_known_as":  "known_as",
        "also_known_as":     "known_as",
        "nickname":          "known_as",
        "alias":             "known_as",
        "published_by":      "produced_by",
        "produced":          "produced_by",
        "executive_produced": "produced_by",
        "co-produced":       "produced_by",
        "wrote":             "written_by",
        "authored":          "authored_by",
        "directed":          "directed_by",
        "co-directed":       "directed_by",
        "starring":          "starred_in",
        "stars":             "starred_in",
        "plays_role_of":     "starred_in",
        "played_role_of":    "starred_in",
        "guest_starred_in":  "appeared_in",
        "appeared_as":       "appeared_in",
        "plays_in":          "appeared_in",
        "played_in":         "appeared_in",
        "released":          "released_in",
        "released_on":       "released_in",
        "published":         "published_in",
        "published_on":      "published_in",
        "founded":           "founded_by",
        "co-founded":        "founded_by",
        "established":       "founded_by",
        "serves":            "serves_as",
        "served_as":         "serves_as",
        "led":               "led_by",
        "leads":             "led_by",
        "manages":           "led_by",
        "coached":           "coached_by",
        "trained_by":        "coached_by",
        "competed_at":       "competed_in",
        "participated_in":   "competed_in",
        "plays_for":         "played_for",
        "represented":       "played_for",
        "is_part_of":        "part_of",
        "is_member_of":      "member_of",
        "is_related_to":     "related_to",
        "has_relation_to":   "related_to",
        "associated_with":   "affiliated_with",
        "collaborated_with": "affiliated_with",
        "worked_with":       "affiliated_with",
        "worked_on":         "affiliated_with",
        "owns":              "affiliated_with",
        "owned_by":          "affiliated_with",
        "operates":          "affiliated_with",
        "located_near":      "located_in",
        "based_in":          "located_in",
        "borders":           "located_in",
        "relocated_to":      "located_in",
        "moved_to":          "located_in",
        "capital":           "capital_of",
        "married":           "married_to",
        "spouse_of":         "married_to",
        "born":              "born_in",
        "died":              "died_in",
        "lost_to":           "involved_in",
        "defeated":          "involved_in",
        "contains":          "has_property",
        "includes":          "has_property",
        "features":          "has_property",
        "consists_of":       "has_property",
        "composed_of":       "has_property",
        "noted_for":         "known_as",
        "is_also_called":    "known_as",
        "also_called":       "known_as",
        "named_after":       "known_as",
        "specializes_in":    "works_as",
        "graduate_of":       "educated_at",
        "attended":          "educated_at",
        "studied_under":     "taught_by",
        "mentored_by":       "taught_by",
        "co-starred":        "appeared_in",
        "co_starred":        "appeared_in",
        "created":           "founded_by",
        "created_by":        "founded_by",
        "replaced_by":       "followed_by",
        "succeeded_by":      "followed_by",
        "succeeded":         "preceded_by",
        "recorded":          "released_in",
        "aired_on":          "released_in",
        "broadcast_on":      "released_in",
        "distributed_by":    "produced_by",
        "distributed":       "produced_by",
    }
    if p in _ALIASES:
        return _ALIASES[p]

    # 3. Token-overlap scoring
    # Stop-tokens are excluded from overlap to avoid spurious matches driven by
    # common words ("for", "a", "of", "in", "by") that carry no semantic content.
    _STOP_TOKENS = frozenset({"a", "an", "the", "of", "for", "in", "by", "at",
                               "to", "is", "was", "be", "been", "as", "on"})
    p_tokens = set(p.split("_")) - _STOP_TOKENS
    best_pred, best_score = "related_to", 0
    for a in allowed:
        a_tokens = set(a.split("_")) - _STOP_TOKENS
        score = len(p_tokens & a_tokens)
        if score > best_score or (score == best_score and len(a) < len(best_pred)):
            best_pred, best_score = a, score

    # Only accept overlap ≥ 1 to avoid nonsense matches
    return best_pred if best_score >= 1 else "related_to"

# --- Predicate whitelist ---
# Why: unconstrained LLMs invent free-text predicates ("is located on the large
# natural bay of", "is also known by her nickname") that fragment the graph into
# disconnected clusters. A shared vocabulary ensures that the same relationship
# extracted from two different documents uses the same predicate string, making
# cross-document entity linking possible.
#
# Design principle: predicates are derived from semantic categories, not from
# observed data. Each category covers a distinct relation class; a predicate is
# added when it is conceptually irreducible to existing ones.
# Source taxonomy: ConceptNet relation types + Wikidata property categories,
# restricted to the domains present in HotpotQA (biography, film/media,
# sport, geography, science, history, politics).
#
# Categories and their predicates:
#
#   [BIOGRAPHICAL — identity & life events]
#     born_in        where a person was born (city/country)
#     born_on        when a person was born (date/year)
#     died_in        where a person died
#     died_on        when a person died
#     nationality    citizenship or ethnic identity of a person
#     known_as       alias, nickname, pen name, stage name
#     works_as       occupation or profession (NEW: distinct from is_a — role, not type)
#
#   [SOCIAL — interpersonal relations]
#     child_of       parent-child (NEW: common bridge in biography questions)
#     married_to     spousal relationship
#     educated_at    institution where a person studied (NEW: academic bridge)
#     taught_by      academic mentor/supervisor (NEW: named-person bridge)
#
#   [CLASSIFICATION — type & membership]
#     is_a           entity type or category
#     part_of        structural component of a larger whole
#     member_of      membership in a group, team, or organisation
#     affiliated_with loose association — when member_of is too strong (NEW)
#
#   [GEOGRAPHY — spatial relations]
#     located_in     physical location of an entity
#     capital_of     a city that is the capital of a country/region (NEW)
#
#   [ORGANISATION — founding & leadership]
#     founded_by     person or group who established an entity
#     founded_in     year or place of founding
#     led_by         current or historical head/leader (NEW: broader than coached_by)
#
#   [CREATIVE WORKS — authorship & production]
#     directed_by    film/theatre director
#     produced_by    film/record producer
#     written_by     author of script, book, article
#     authored_by    book author (distinct from written_by: longer-form works)
#     composed_by    musical composer (NEW: distinct from written_by)
#     based_on       adaptation source — novel, true story, earlier work (NEW)
#     appeared_in    participation broader than a lead role (NEW: vs. starred_in)
#     starred_in     lead acting role
#     released_in    publication or release year/location
#     published_in   for books, journals, periodicals (NEW: distinct from released_in)
#
#   [AWARDS & RECOGNITION]
#     won            award or title won
#     nominated_for  award nomination without win
#
#   [SPORT]
#     played_for     team or club a player represented
#     coached_by     coach or manager of a team/athlete
#     competed_in    event or competition participated in (NEW)
#
#   [SEQUENTIAL — ordering & succession]
#     followed_by    successor (person, work, event)
#     preceded_by    predecessor
#
#   [ROLES — contextual assignments]
#     serves_as      role held by a person in an organisation or context (NEW)
#
#   [FALLBACK — when no specific predicate fits]
#     Use these only if no specific predicate above applies.
#     Ensures that entities are still added to the graph (and reachable via BFS)
#     even when the exact relationship type cannot be expressed precisely.
#     Ordered from most to least specific:
#     has_property   entity has an attribute or characteristic
#                    (e.g. "Novel X has_property Genre Mystery")
#     involved_in    entity participates in an event, process or situation
#                    (e.g. "Person X involved_in Conflict Y")
#     related_to     last-resort generic link — use only when nothing else fits
#
_ALLOWED_PREDICATES = ", ".join([
    # Biographical
    "born_in", "born_on", "died_in", "died_on", "nationality", "known_as", "works_as",
    # Social
    "child_of", "married_to", "educated_at", "taught_by",
    # Classification
    "is_a", "part_of", "member_of", "affiliated_with",
    # Geography
    "located_in", "capital_of",
    # Organisation
    "founded_by", "founded_in", "led_by",
    # Creative works
    "directed_by", "produced_by", "written_by", "authored_by",
    "composed_by", "based_on", "appeared_in", "starred_in", "released_in", "published_in",
    # Awards
    "won", "nominated_for",
    # Sport
    "played_for", "coached_by", "competed_in",
    # Sequential
    "followed_by", "preceded_by",
    # Roles
    "serves_as",
    # Fallback (use only when no specific predicate fits)
    "has_property", "involved_in", "related_to",
])

# Frozenset for O(1) membership tests in _normalize_predicate
_ALLOWED_PREDICATES_SET: frozenset[str] = frozenset(
    p.strip() for p in _ALLOWED_PREDICATES.split(",")
)

_SYSTEM_PROMPT = (
    "Extract the most important entities and relationships from the text as triples. "
    "Return a JSON object with a 'triples' key containing an array of at most 20 objects, "
    "each with 'subject', 'predicate', and 'object' string fields.\n"
    "Rules:\n"
    "  - subject and object: 1-5 words, Title Case (e.g. 'Clara Voss', 'Lake Arven').\n"
    f"  - predicate: snake_case, choose the closest specific match from: {_ALLOWED_PREDICATES}.\n"
    "    Use has_property / involved_in / related_to only as last resort — prefer specific predicates.\n"
    "  - NEVER put a list in object. One entity per triple.\n"
    "    Bad:  {\"subject\": \"Film X\", \"predicate\": \"starred_in\", "
    "\"object\": \"Actor A, Actor B, Actor C\"}\n"
    "    Good: three separate triples, one per actor.\n"
    'Return {"triples": []} if no clear relationships exist. '
    "Output only the JSON object, no other text."
)

TRIPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject":   {"type": "string"},
                    "predicate": {"type": "string"},
                    "object":    {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["triples"],
    "additionalProperties": False,
}

# Fallback: extract triples from truncated JSON via regex
_TRIPLE_RE = re.compile(
    r'\{\s*"subject"\s*:\s*"(?P<s>[^"]+)"\s*,\s*"predicate"\s*:\s*"(?P<p>[^"]+)"\s*,'
    r'\s*"object"\s*:\s*"(?P<o>[^"]+)"\s*\}'
)


def _split_list_objects(triples: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Post-processing fallback: if the LLM returns a comma-separated list as
    the object despite the prompt instruction, split it into individual triples.
    Heuristic: split only when ≥3 comma-separated parts each ≤40 chars.
    Short pairs like "New York, USA" (2 parts) are kept intact.
    """
    result: list[dict[str, str]] = []
    for t in triples:
        parts = [p.strip() for p in t["object"].split(",") if p.strip()]
        if len(parts) >= 3 and all(len(p) <= 40 for p in parts):
            for part in parts:
                result.append({"subject": t["subject"], "predicate": t["predicate"], "object": part})
        else:
            result.append(t)
    return result


def parse_triples(raw: str) -> list[dict[str, str]]:
    """Parse an LLM response into normalised (subject, predicate, object) triples.

    Shared by both graph variants (GraphMemory and LlamaIndexGraphMemory) so they
    apply identical parsing: JSON-first with a regex fallback for truncated output,
    list-object splitting, and predicate normalisation against the shared whitelist.
    Keeps entity casing verbatim (unlike LlamaIndex's default parser, which
    lowercases via .capitalize() and drops comma-valued objects entirely).
    """
    raw = raw or "{}"
    try:
        parsed = json.loads(raw)
        triples = parsed.get("triples", []) if isinstance(parsed, dict) else []
    except json.JSONDecodeError:
        matches = list(_TRIPLE_RE.finditer(raw))
        if matches:
            logger.debug("parse_triples: JSON truncated, recovered %d triples via regex", len(matches))
            triples = [{"subject": m.group("s"), "predicate": m.group("p"), "object": m.group("o")} for m in matches]
        else:
            logger.warning("parse_triples: JSON parse failed: %r", raw[:200])
            return []

    valid = [
        t for t in triples
        if isinstance(t, dict)
        and all(isinstance(t.get(k), str) and t.get(k, "").strip() for k in ("subject", "predicate", "object"))
    ]
    split = _split_list_objects(valid)

    # Normalise predicates: map LLM-invented free-text predicates to the
    # nearest entry in _ALLOWED_PREDICATES_SET so the graph stays clean.
    normalised = []
    for t in split:
        raw_p = t["predicate"]
        norm_p = _normalize_predicate(raw_p, _ALLOWED_PREDICATES_SET)
        if norm_p != raw_p:
            logger.debug("parse_triples: predicate %r → %r", raw_p, norm_p)
        normalised.append({**t, "predicate": norm_p})

    logger.debug(
        "parse_triples: %d valid → %d after list-split → %d after normalisation",
        len(valid), len(split), len(normalised),
    )
    return normalised
