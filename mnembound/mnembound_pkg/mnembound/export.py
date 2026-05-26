"""
mnembound.export — Safe view-restricted exports for downstream consumers.

The Corpus dataclass intentionally holds both fragment-level data (the
records and fragments themselves) and store-level data (the salt, the
ProvenanceResolver, the tenant entity pools, the entity sanitizer).
Passing the whole Corpus to a fragment-level baseline would break the
§4.2 abstraction by giving the baseline access to information that
should be exclusive to store-level verifiers.

This module provides explicit export functions, one per consumer type.
Each function returns a narrowly-typed bundle containing ONLY what that
consumer is allowed to see. There is no `export_everything()` because
there is no consumer that should see everything.

Three exports:

- export_fragment_view:  for V_frag baselines (citation logging,
                         faithfulness, NLI, source-set entailment, graph
                         consistency). Returns ONLY fragment-level views
                         (role, value, text, opaque provenance). Does NOT
                         include the resolver, salt, sanitizer, or any
                         hidden record/boundary/event metadata. Caller
                         picks tenant_visible=True or False explicitly.

- export_bcesv_view:     for BCESV (the defense). Returns fragments with
                         the resolver and a per-fragment (boundary_id,
                         event_id) lookup. BCESV-Exact uses the lookup
                         directly. BCESV-Reconstruct additionally gets
                         tenant_id, timestamp, log_source, incident_
                         template, and entity_set per fragment.

- export_store_view:     for ground-truth scoring and validation. Returns
                         everything: full records, full fragments,
                         resolver, sanitizer, tenant pools. Used by the
                         scoring module (which decides per-recall whether
                         it is event_supported, fragment_supported, etc.)
                         and by the realism audit.

Tests in test_export_safety.py verify that each export's keys never
include forbidden fields and that nothing accidentally pulls the resolver
into the V_frag bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from mnembound.schema import Corpus, Fragment, ProvenanceResolver, Record


# Keys that must NEVER appear in a fragment-level export. If any of these
# keys is found in a FragmentViewBundle entry, the export is broken and
# the §4.2 abstraction is violated.
FORBIDDEN_KEYS_IN_FRAGMENT_VIEW: frozenset[str] = frozenset({
    "record_id",
    "boundary_id",
    "event_id",
    "tenant_id",
    "timestamp",
    "log_source",
    "incident_template",
    "entity_set",
    "salt",
    "resolver",
    "internal_provenance",
})


# ----- Fragment-level export (for V_frag baselines) -----


@dataclass(frozen=True)
class FragmentViewBundle:
    """
    The export bundle for V_frag baselines.

    Contains a tuple of per-fragment view dicts and NOTHING ELSE. The bundle
    deliberately does not carry the salt, the resolver, the entity sanitizer,
    or any tenant/record/boundary/event metadata. If you find yourself
    wanting to read those from a FragmentViewBundle, you are operating at
    the wrong level: switch to export_bcesv_view or export_store_view.
    """

    tenant_visible: bool
    fragments: tuple[Mapping[str, Any], ...]


def export_fragment_view(
    corpus: Corpus, *, tenant_visible: bool
) -> FragmentViewBundle:
    """
    Export the V_frag-visible view of every fragment in the corpus.

    Args:
        corpus: The full corpus.
        tenant_visible: REQUIRED keyword-only. Whether to include tenant
            identity lexically in `value` and `text`. The default
            experimental condition in the paper is tenant_visible=False;
            the tenant-visible setting is used only for the §4.4
            source-set-entailment fairness baseline.

    Returns:
        FragmentViewBundle whose `fragments` tuple contains exactly the
        per-fragment view dicts. Each dict has keys
        {role, value, text, provenance}.
    """
    views = tuple(
        MappingProxyType(f.view_for_v_frag(tenant_visible=tenant_visible))
        for f in corpus.fragments
    )
    return FragmentViewBundle(tenant_visible=tenant_visible, fragments=views)


# ----- BCESV export (for the defense) -----


@dataclass(frozen=True)
class BcesvFragmentRecord:
    """
    One fragment as visible to BCESV.

    BCESV-Exact uses (boundary_id, event_id) directly.
    BCESV-Reconstruct additionally uses (tenant_id, timestamp, log_source,
    incident_template, entity_set).

    The opaque public_provenance_id is included so BCESV can join its
    output back against retrieval traces and audit logs that key on opaque
    IDs. The internal ProvenanceHandle is NOT exposed; BCESV does not need
    record_id directly (although it can resolve it via the resolver if
    needed for any debugging).
    """

    public_provenance_id: str
    role: str
    value: str  # tenant-visible (BCESV is a store-level verifier; visibility is fine)
    text: str

    # The hidden metadata BCESV needs.
    boundary_id: str
    event_id: str
    tenant_id: str
    timestamp: str
    log_source: str
    incident_template: str
    entity_set: frozenset[str]


@dataclass(frozen=True)
class BcesvBundle:
    """Bundle for BCESV. Includes the resolver for ID-to-record lookup."""

    fragments: tuple[BcesvFragmentRecord, ...]
    resolver: ProvenanceResolver
    # Fast lookup from public_provenance_id -> BcesvFragmentRecord for
    # BCESV implementations that prefer dict access.
    fragments_by_public_id: Mapping[str, BcesvFragmentRecord]


def export_bcesv_view(corpus: Corpus) -> BcesvBundle:
    """
    Export the BCESV-visible view of the corpus.

    BCESV is a store-level verifier and falls outside the §4.2 abstraction
    by design. It receives full per-fragment metadata (boundary, event,
    tenant, timestamp, log source, incident template, entity set) plus the
    resolver for opaque-ID-to-record lookup.

    What BCESV does NOT receive: the entity sanitizer (BCESV has no need
    for sanitized strings; that's a V_frag concern) and the full Corpus
    object (we hand BCESV a narrowly-typed bundle to make accidental
    leakage harder).
    """
    if corpus.resolver is None:
        raise ValueError(
            "Corpus has no resolver; cannot export BCESV view. The generator "
            "always builds a resolver, so this indicates the corpus was "
            "constructed manually without one."
        )

    bcesv_fragments = tuple(
        BcesvFragmentRecord(
            public_provenance_id=f.public_provenance_id,
            role=f.role.value,
            value=f.value,
            text=f.text,
            boundary_id=f.boundary_id,
            event_id=f.event_id,
            tenant_id=f.tenant_id,
            timestamp=f.timestamp,
            log_source=f.log_source,
            incident_template=f.incident_template,
            entity_set=f.entity_set,
        )
        for f in corpus.fragments
    )

    by_id: dict[str, BcesvFragmentRecord] = {
        bf.public_provenance_id: bf for bf in bcesv_fragments
    }

    return BcesvBundle(
        fragments=bcesv_fragments,
        resolver=corpus.resolver,
        fragments_by_public_id=MappingProxyType(by_id),
    )


# ----- Store-level export (for scoring and validation) -----


@dataclass(frozen=True)
class StoreViewBundle:
    """
    Full store-level view of the corpus.

    Used by:
    - The scoring module (computes is_event_supported, is_fragment_supported,
      boundary-invalid recall rate, etc. against construction-known
      ground truth).
    - The validation module (already imports Corpus directly).
    - The realism audit.

    This bundle is essentially a typed wrapper around the corpus. We expose
    it as a separate function so that callers must state "I want store-level
    access" explicitly, which is a useful signal in code review.
    """

    records: tuple[Record, ...]
    fragments: tuple[Fragment, ...]
    resolver: ProvenanceResolver
    tenant_entity_pools: Mapping[str, frozenset[str]]
    entity_sanitizer: Mapping[str, str]
    salt: str


def export_store_view(corpus: Corpus) -> StoreViewBundle:
    """
    Export the full store-level view.

    Caller must hold this only for scoring, validation, realism audit, or
    other ground-truth-bearing code. Never pass this object or any of its
    fields to a V_frag baseline.
    """
    if corpus.resolver is None:
        raise ValueError(
            "Corpus has no resolver; cannot export store view."
        )
    return StoreViewBundle(
        records=corpus.records,
        fragments=corpus.fragments,
        resolver=corpus.resolver,
        tenant_entity_pools=corpus.tenant_entity_pools,
        entity_sanitizer=corpus.entity_sanitizer,
        salt=corpus.salt,
    )


# ----- Validation helper -----


def assert_fragment_view_is_clean(bundle: FragmentViewBundle) -> None:
    """
    Assert that a FragmentViewBundle contains no forbidden keys in any
    fragment view. Raises AssertionError on violation.

    Test code uses this to verify abstraction integrity.
    """
    for i, view in enumerate(bundle.fragments):
        leaked = set(view.keys()) & FORBIDDEN_KEYS_IN_FRAGMENT_VIEW
        assert not leaked, (
            f"FragmentViewBundle entry {i} exposes forbidden keys: "
            f"{sorted(leaked)}"
        )
