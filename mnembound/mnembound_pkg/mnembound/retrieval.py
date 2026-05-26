"""
mnembound.retrieval — Retrieval and aggregation rungs.

The paper evaluates four memory-sharing rungs that increase the risk of
event-level provenance failure:

1. ISOLATED                      — per-tenant memory; no cross-tenant fragments
2. BOUNDARY_PRESERVING_TOPK      — shared index but boundary-filtered retrieval
3. SHARED_VECTOR_INDEX           — single shared index; top-k by similarity
4. SUMMARY_MERGE                 — per-tenant summaries merged into agent context

Each rung is implemented as a callable that takes a Probe and a Corpus and
returns a `RetrievalResult`. The result carries:
- `fragments`: the fragments the agent will see (the retrieval set)
- `context_text`: a rendered text bundle for prompting (used only by
  summary_merge, which produces a unified summary)
- `rung_name`: identifies the rung for logging and scoring

Strict vs dynamic retrieval
---------------------------
Two retrieval modes are exposed:

- STRICT (default): the rung filters the probe's pre-attached fragments
  according to its sharing semantics. This is what the §7 deterministic
  sweep uses, because it preserves the construction-time ground truth
  labels exactly. ISOLATED and BOUNDARY_PRESERVING_TOPK can hide
  cross-tenant fragments; SHARED_VECTOR_INDEX returns all fragments;
  SUMMARY_MERGE concatenates them into context_text.

- DYNAMIC (Day 7+, scaffolded but not implemented): the rung issues the
  probe's `query` against a corpus index and returns top-k. This is what
  the §9 bounded LLM run uses. Real embedding-based dynamic retrieval is left as future work; the current rungs use
  retrieval indexes (BM25, sentence-embedding). For now, all rungs
  operate in strict mode.

Tenant context
--------------
Every probe must declare a "querying tenant" — the tenant on whose
behalf the agent is running. For SUPPORTED/ABSENT/PARTIAL probes this is
unambiguous (the tenant of the underlying record). For CROSS_BOUNDARY
traps the querying tenant is the tenant of the probe's *recall* —
specifically, the tenant of whichever source event contributed the
subject. This is documented in the rung implementations and verified by
test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from mnembound.schema import (
    ALL_ROLES,
    ComponentRole,
    Corpus,
    Fragment,
    Probe,
    ProbeType,
)


class Rung(str, Enum):
    """The four sharing rungs."""

    ISOLATED = "isolated"
    BOUNDARY_PRESERVING_TOPK = "boundary_preserving_topk"
    SHARED_VECTOR_INDEX = "shared_vector_index"
    SUMMARY_MERGE = "summary_merge"


ALL_RUNGS: tuple[Rung, ...] = (
    Rung.ISOLATED,
    Rung.BOUNDARY_PRESERVING_TOPK,
    Rung.SHARED_VECTOR_INDEX,
    Rung.SUMMARY_MERGE,
)


@dataclass(frozen=True)
class RetrievalResult:
    """
    The output of one retrieval call.

    Fields:
        rung: which rung produced this result
        fragments: the fragments the agent sees (post-filtering)
        querying_tenant: the tenant on whose behalf the agent is querying
        context_text: optional unified text for SUMMARY_MERGE; empty for
            the other rungs (they expose individual fragments).
        excluded_fragments: fragments that WERE in the probe's pre-attached
            set but were filtered out by the rung. For ISOLATED, these are
            the cross-tenant fragments. For BOUNDARY_PRESERVING_TOPK, the
            cross-boundary ones. SHARED_VECTOR_INDEX and SUMMARY_MERGE
            exclude nothing. Useful for logging and ablation analysis.
    """

    rung: Rung
    fragments: tuple[Fragment, ...]
    querying_tenant: str
    context_text: str = ""
    excluded_fragments: tuple[Fragment, ...] = field(default_factory=tuple)


# ----- Querying tenant resolution -----


def querying_tenant_for(probe: Probe) -> str:
    """
    Determine which tenant is querying for this probe.

    For SUPPORTED/PARTIAL: the tenant of the underlying record. We recover
    this from the supporting fragments' tenant_id (which is consistent for
    these probe types).

    For ABSENT: the tenant of the base record from which the recall was
    derived. We recover this from the subject fragment's value if any
    fragment has matching role.subject and value == recall.subject —
    otherwise we fall back to the first retrieved fragment's tenant.

    For CROSS_BOUNDARY: the tenant of the source event that contributed
    the subject component. This is the agent's "context" tenant; the
    other source events' tenants are the ones the cross-boundary trap
    inappropriately pulls in.

    The resolution is deterministic given a probe.
    """
    if not probe.fragments:
        raise ValueError(
            f"probe {probe.probe_id} has no fragments; cannot determine "
            "querying tenant"
        )

    # For SUPPORTED/PARTIAL, all fragments share a tenant.
    if probe.probe_type in (ProbeType.SUPPORTED, ProbeType.PARTIAL):
        tenants = {f.tenant_id for f in probe.fragments}
        # SUPPORTED with distractors may have multiple tenants among
        # distractors; the canonical querying tenant is the underlying
        # event's tenant. We recover it by finding the fragment whose
        # (boundary, event) pair appears in probe.boundary_event_pairs.
        if probe.boundary_event_pairs:
            # The underlying event's (boundary, event) is in the set.
            target_pair = next(iter(probe.boundary_event_pairs))
            for f in probe.fragments:
                if (f.boundary_id, f.event_id) == target_pair:
                    return f.tenant_id
        # Fallback: first fragment's tenant.
        return probe.fragments[0].tenant_id

    # For CROSS_BOUNDARY, the subject fragment's tenant is the querying
    # tenant by convention. This matches the paper's framing: an agent
    # asks "what happened with subject X?" from the tenant where X lives.
    if probe.probe_type == ProbeType.CROSS_BOUNDARY:
        for f in probe.fragments:
            if f.role == ComponentRole.SUBJECT:
                return f.tenant_id
        return probe.fragments[0].tenant_id

    # For ABSENT, find the fragment whose value matches the recall's
    # subject (this is the base-record fragment from which the recall was
    # derived). If none match (subject was the swapped component), fall
    # back to the first fragment.
    if probe.probe_type == ProbeType.ABSENT:
        for f in probe.fragments:
            if f.role == ComponentRole.SUBJECT and f.value == probe.recall.subject:
                return f.tenant_id
        # The subject was the swapped component, so no fragment carries
        # it. Pick the tenant that contributes the most non-subject
        # fragments (the base record's tenant).
        tenant_counts: dict[str, int] = {}
        for f in probe.fragments:
            tenant_counts[f.tenant_id] = tenant_counts.get(f.tenant_id, 0) + 1
        return max(tenant_counts.items(), key=lambda kv: kv[1])[0]

    raise ValueError(f"unknown probe type: {probe.probe_type}")


# ----- Rung implementations (strict mode) -----


def retrieve_isolated(probe: Probe, corpus: Corpus) -> RetrievalResult:
    """
    ISOLATED rung: only fragments from the querying tenant are visible.

    Cross-tenant fragments are filtered out. For SUPPORTED probes, this
    is a no-op (all fragments are same-tenant). For CROSS_BOUNDARY
    traps, this strips the cross-tenant fragments, which means the
    composed recall is no longer fragment-supported — exactly the
    isolation guarantee.
    """
    tenant = querying_tenant_for(probe)
    keep: list[Fragment] = []
    drop: list[Fragment] = []
    for f in probe.fragments:
        if f.tenant_id == tenant:
            keep.append(f)
        else:
            drop.append(f)
    return RetrievalResult(
        rung=Rung.ISOLATED,
        fragments=tuple(keep),
        querying_tenant=tenant,
        excluded_fragments=tuple(drop),
    )


def retrieve_boundary_preserving_topk(
    probe: Probe, corpus: Corpus
) -> RetrievalResult:
    """
    BOUNDARY_PRESERVING_TOPK rung: shared index, but retrieval is
    filtered to the querying tenant's boundary.

    Under TENANT boundary granularity, this is equivalent to ISOLATED
    (one boundary per tenant). Under ENGAGEMENT or EVENT granularity,
    this is stricter: only fragments from the SAME boundary as the
    querying tenant's "current" boundary are visible. For now we
    interpret "current boundary" as the boundary of the subject fragment
    (or the first fragment if no subject is present).
    """
    tenant = querying_tenant_for(probe)
    # Determine the querying boundary: the boundary of the subject
    # fragment if any, else the first fragment.
    querying_boundary = None
    for f in probe.fragments:
        if f.role == ComponentRole.SUBJECT and f.tenant_id == tenant:
            querying_boundary = f.boundary_id
            break
    if querying_boundary is None:
        for f in probe.fragments:
            if f.tenant_id == tenant:
                querying_boundary = f.boundary_id
                break
    if querying_boundary is None and probe.fragments:
        querying_boundary = probe.fragments[0].boundary_id

    keep: list[Fragment] = []
    drop: list[Fragment] = []
    for f in probe.fragments:
        if f.boundary_id == querying_boundary:
            keep.append(f)
        else:
            drop.append(f)
    return RetrievalResult(
        rung=Rung.BOUNDARY_PRESERVING_TOPK,
        fragments=tuple(keep),
        querying_tenant=tenant,
        excluded_fragments=tuple(drop),
    )


def retrieve_shared_vector_index(
    probe: Probe, corpus: Corpus
) -> RetrievalResult:
    """
    SHARED_VECTOR_INDEX rung: all fragments are visible regardless of
    tenant or boundary.

    In strict mode, this returns the probe's pre-attached fragments
    unchanged. In dynamic mode (Day 7+), this will execute a top-k
    similarity search against the corpus index.

    This is the rung where cross-boundary traps fire: the agent receives
    fragments from multiple tenants and composes them.
    """
    tenant = querying_tenant_for(probe)
    return RetrievalResult(
        rung=Rung.SHARED_VECTOR_INDEX,
        fragments=probe.fragments,
        querying_tenant=tenant,
    )


def retrieve_summary_merge(probe: Probe, corpus: Corpus) -> RetrievalResult:
    """
    SUMMARY_MERGE rung: per-tenant summaries are concatenated into a
    unified context.

    All retrieved fragments are kept; the agent will receive them as a
    merged summary at prompt-construction time. We deliberately do NOT
    populate context_text here, because doing so would force a choice
    of tenant_visible vs tenant_hidden at retrieval time and could leak
    tenant-revealing text into downstream logs. The prompt builder calls
    `render_summary_text(probe, tenant_visible=...)` at the point of use,
    which makes the visibility choice explicit and local.

    Why this is the highest-risk rung: the agent loses per-fragment
    provenance entirely. There is no way to point at a specific source
    fragment; the agent must produce a recall from a soup of tenant-
    mixed text. Cross-boundary recall is structurally encouraged.

    The `fragments` field is populated (with all retrieved fragments) so
    scoring predicates can run; the prompt rendering happens later.
    """
    tenant = querying_tenant_for(probe)

    return RetrievalResult(
        rung=Rung.SUMMARY_MERGE,
        fragments=probe.fragments,
        querying_tenant=tenant,
        context_text="",  # rendered on demand by render_summary_text()
    )


class SummaryMergeMode(str, Enum):
    """
    How the summary_merge rung renders its evidence text.

    PER_CLIENT_BLOCKS: groups fragments by their source tenant and renders
        one "Summary of recent activity for this client:" block per tenant.
        Even in tenant-hidden mode, the multi-block structure signals to the
        agent that the evidence comes from distinct accounts, which can
        suppress cross-boundary composition. This is the "safer aggregator"
        version.

    COLLAPSED: dumps every retrieved fragment's text into a single bulleted
        list with no per-tenant grouping ("Merged memory summary:\\n- ...").
        This is the closest analogue to a real aggregator that has merged
        evidence into a single context window. Recommended for the headline
        summary_merge condition in the paper.

    Default in the paper headline: COLLAPSED.
    """

    PER_CLIENT_BLOCKS = "per_client_blocks"
    COLLAPSED = "collapsed"


def render_summary_text(
    probe: Probe,
    *,
    tenant_visible: bool,
    mode: SummaryMergeMode = SummaryMergeMode.COLLAPSED,
) -> str:
    """
    Render the summary_merge context text body (without an outer label
    header; the prompt builder adds that).

    Args:
        probe: The probe whose retrieved fragments are summarized.
        tenant_visible: REQUIRED keyword. If False, uses `f.sanitized_text`;
            otherwise `f.text`.
        mode: PER_CLIENT_BLOCKS or COLLAPSED. See SummaryMergeMode.
    """
    def fragment_text(f: Fragment) -> str:
        return f.text if tenant_visible else f.sanitized_text

    if mode == SummaryMergeMode.COLLAPSED:
        # Single flat bullet list. Sort by (tenant, role) for determinism
        # but expose no per-tenant grouping in the output.
        sorted_frags = sorted(
            probe.fragments,
            key=lambda f: (f.tenant_id, f.role.value),
        )
        return "\n".join(f"- {fragment_text(f)}" for f in sorted_frags)

    # PER_CLIENT_BLOCKS
    by_tenant: dict[str, list[Fragment]] = {}
    for f in probe.fragments:
        by_tenant.setdefault(f.tenant_id, []).append(f)
    summary_parts: list[str] = []
    for tid in sorted(by_tenant.keys()):
        frags = sorted(by_tenant[tid], key=lambda f: f.role.value)
        bullet_lines = [f"  - {fragment_text(f)}" for f in frags]
        summary_parts.append(
            "Summary of recent activity for this client:\n" + "\n".join(bullet_lines)
        )
    return "\n\n".join(summary_parts)


# ----- Dispatch -----


_RETRIEVAL_FUNCTIONS: dict[Rung, Callable[[Probe, Corpus], RetrievalResult]] = {
    Rung.ISOLATED: retrieve_isolated,
    Rung.BOUNDARY_PRESERVING_TOPK: retrieve_boundary_preserving_topk,
    Rung.SHARED_VECTOR_INDEX: retrieve_shared_vector_index,
    Rung.SUMMARY_MERGE: retrieve_summary_merge,
}


def retrieve(rung: Rung, probe: Probe, corpus: Corpus) -> RetrievalResult:
    """Dispatch to the appropriate rung's retrieval function."""
    fn = _RETRIEVAL_FUNCTIONS.get(rung)
    if fn is None:
        raise ValueError(f"unknown rung: {rung}")
    return fn(probe, corpus)


# ----- Per-rung statistics -----


@dataclass(frozen=True)
class RungSummary:
    """Aggregate stats over one rung's retrievals across many probes."""

    rung: Rung
    n_probes: int
    avg_fragments_retrieved: float
    avg_fragments_excluded: float
    # Per probe type, the fraction of fragments retained vs probe.fragments.
    retention_by_probe_type: dict[str, float]


def summarize_rung(
    rung: Rung, probes: list[Probe], corpus: Corpus
) -> RungSummary:
    """Run a rung over a list of probes and report aggregate statistics."""
    if not probes:
        return RungSummary(
            rung=rung,
            n_probes=0,
            avg_fragments_retrieved=0.0,
            avg_fragments_excluded=0.0,
            retention_by_probe_type={},
        )

    total_retrieved = 0
    total_excluded = 0
    retained_by_type: dict[str, list[float]] = {}

    for p in probes:
        result = retrieve(rung, p, corpus)
        total_retrieved += len(result.fragments)
        total_excluded += len(result.excluded_fragments)
        total_in = len(p.fragments)
        retention = len(result.fragments) / total_in if total_in > 0 else 0.0
        retained_by_type.setdefault(p.probe_type.value, []).append(retention)

    return RungSummary(
        rung=rung,
        n_probes=len(probes),
        avg_fragments_retrieved=total_retrieved / len(probes),
        avg_fragments_excluded=total_excluded / len(probes),
        retention_by_probe_type={
            t: sum(v) / len(v) for t, v in retained_by_type.items()
        },
    )
