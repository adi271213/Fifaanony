"""
mnembound.schema — Data structures for SOC records, fragments, and probes.

These dataclasses mirror the formal model from sections 3 and 4 of the paper:

    m = (client_id, boundary_id, event_id, subject, action, object_, outcome, time, provenance)

A `Record` is a full SOC event with all components. A `Fragment` is one
component of a record exposed at retrieval time; the per-component grain
matches the proof construction in §4.3.

- PUBLIC_HANDLE_HEX_LEN raised from 16 to 24 (96-bit) so reviewers do not
  flag truncated-SHA-256 concerns.
- Fragment stores BOTH tenant-visible and tenant-hidden (sanitized) forms
  of `value` and `text`. The view_for_v_frag() method takes a
  `tenant_visible` parameter (default True). When False, the view exposes
  the sanitized forms, matching the §4.4 baseline-fairness setup.
- ProvenanceResolver is unchanged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class ComponentRole(str, Enum):
    """The five components of a SOC recall (subject, action, object, outcome, time)."""

    SUBJECT = "subject"
    ACTION = "action"
    OBJECT = "object"
    OUTCOME = "outcome"
    TIME = "time"


ALL_ROLES: tuple[ComponentRole, ...] = (
    ComponentRole.SUBJECT,
    ComponentRole.ACTION,
    ComponentRole.OBJECT,
    ComponentRole.OUTCOME,
    ComponentRole.TIME,
)


# Length (hex characters) of the public opaque provenance ID.
# 24 hex chars = 96 bits of entropy. At 250k fragments, the birthday-bound
# collision probability is ~10^-18; we also run an explicit collision check
# in the generator.
PUBLIC_HANDLE_HEX_LEN = 24


@dataclass(frozen=True)
class ProvenanceHandle:
    """
    Opaque authenticity and retrieval-membership evidence (§4.2).

    The HIDDEN form carries (record_id, fragment_index) for internal use by
    the generator and by store-level verifiers via ProvenanceResolver. The
    PUBLIC form, which is what V_frag baselines see, is a salted SHA-256
    digest that exposes neither the record nor the fragment index.
    """

    record_id: str
    fragment_index: int

    def public_id(self, salt: str) -> str:
        """Return the V_frag-visible opaque handle."""
        raw = f"{salt}|{self.record_id}|{self.fragment_index}".encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()[:PUBLIC_HANDLE_HEX_LEN]
        return f"prov:{digest}"


@dataclass(frozen=True)
class ProvenanceResolver:
    """
    Resolves opaque public provenance IDs back to records.

    Granted only to store-level verifiers (BCESV-Exact, BCESV-Reconstruct,
    and the tenant-visible source-set entailment baseline per §4.4). NEVER
    passed to V_frag baselines.

    Possession of a ProvenanceResolver moves a verifier outside the
    fragment-level abstraction.
    """

    salt: str
    _public_to_internal: Mapping[str, tuple[str, int]]

    def resolve(self, public_id: str) -> tuple[str, int]:
        return self._public_to_internal[public_id]

    def __contains__(self, public_id: str) -> bool:
        return public_id in self._public_to_internal


def make_resolver(
    salt: str, public_to_internal: dict[str, tuple[str, int]]
) -> ProvenanceResolver:
    """Construct a ProvenanceResolver with an immutable view of the mapping."""
    return ProvenanceResolver(
        salt=salt,
        _public_to_internal=MappingProxyType(dict(public_to_internal)),
    )


@dataclass(frozen=True)
class Record:
    """
    A complete SOC incident record.

    Invariants enforced by the generator:
    - (boundary_id, event_id) is unique across the entire corpus.
    - boundary_id is in the per-tenant disjoint namespace for client_id.
    - All five component values are drawn from the per-tenant entity pools
      (Day 2). Day 3 introduces explicit bridge entities tracked separately.
    """

    record_id: str
    client_id: str
    boundary_id: str
    event_id: str

    subject: str
    action: str
    object_: str
    outcome: str
    time: str

    log_source: str
    incident_template: str

    component_text: Mapping[ComponentRole, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    extra: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def component(self, role: ComponentRole) -> str:
        if role is ComponentRole.SUBJECT:
            return self.subject
        if role is ComponentRole.ACTION:
            return self.action
        if role is ComponentRole.OBJECT:
            return self.object_
        if role is ComponentRole.OUTCOME:
            return self.outcome
        if role is ComponentRole.TIME:
            return self.time
        raise ValueError(f"unknown role: {role}")

    def text_for(self, role: ComponentRole) -> str:
        return self.component_text[role]

    def boundary_event_key(self) -> tuple[str, str]:
        return (self.boundary_id, self.event_id)


@dataclass(frozen=True)
class Fragment:
    """
    A per-component fragment exposed at retrieval time.

    Two views are available:

    - **tenant-visible**:    `value` and `text` carry the original entity
                             strings (e.g., 'host:web001-A.clientA.local').
                             Used in the §7 tenant-visible source-set
                             entailment baseline and for store-level
                             verifiers.
    - **tenant-hidden**:     `sanitized_value` and `sanitized_text` are the
                             entity-type-preserving but tenant-identity-
                             stripped forms (e.g., 'host:h_a8f3c12b'). Used
                             in the §7 default tenant-hidden baselines.

    view_for_v_frag(tenant_visible=...) returns the appropriate view. In
    both views, the provenance field is the opaque public_provenance_id,
    and the keys are exactly {role, value, text, provenance}.

    The hidden fields are stored on the Fragment dataclass for generator
    convenience, but view_for_v_frag never includes them in any mode.
    """

    role: ComponentRole

    # Tenant-visible component value (default view).
    value: str
    # Tenant-hidden, type-preserving sanitized form of `value`.
    sanitized_value: str

    # Tenant-visible per-component sentence text.
    text: str
    # Tenant-hidden sanitized form of `text`.
    sanitized_text: str

    # Opaque public ID (V_frag-visible regardless of tenant mode).
    public_provenance_id: str

    # Internal handle. Used by the generator and store-level verifiers.
    # NOT exposed via view_for_v_frag.
    provenance: ProvenanceHandle

    # Hidden from V_frag. Exposed only to store-level verifiers.
    record_id: str
    boundary_id: str
    event_id: str
    tenant_id: str
    timestamp: str
    log_source: str
    incident_template: str
    entity_set: frozenset[str]

    def view_for_v_frag(self, *, tenant_visible: bool) -> dict[str, Any]:
        """
        Return the fragment-level view (§4.2).

        The `tenant_visible` argument is keyword-only and has NO default.
        This is deliberate: forgetting to specify tenant visibility is a
        common bug that silently corrupts the audit-invisibility result by
        leaking tenant identity into baselines that should not see it.
        Callers MUST choose explicitly.

        When tenant_visible=True, value and text carry the original entity
        strings, which contain tenant identity lexically. Use this view for
        the §7 tenant-visible source-set entailment baseline.

        When tenant_visible=False, value and text are sanitized: entity
        strings are replaced with tenant-hidden opaque tokens that preserve
        entity type. Use this view for the §7 tenant-hidden baselines, which
        are the common deployment surface.

        In BOTH modes, the view contains exactly
        {role, value, text, provenance} and never includes tenant_id,
        record_id, boundary_id, event_id, timestamp, log_source,
        incident_template, or entity_set.

        Opacity is verified mechanically in tests/test_provenance_opacity.py.
        """
        if tenant_visible:
            return {
                "role": self.role.value,
                "value": self.value,
                "text": self.text,
                "provenance": self.public_provenance_id,
            }
        return {
            "role": self.role.value,
            "value": self.sanitized_value,
            "text": self.sanitized_text,
            "provenance": self.public_provenance_id,
        }


@dataclass(frozen=True)
class Recall:
    """An event description produced by the agent or constructed for experiments."""

    subject: str
    action: str
    object_: str
    outcome: str
    time: str

    def component(self, role: ComponentRole) -> str:
        if role is ComponentRole.SUBJECT:
            return self.subject
        if role is ComponentRole.ACTION:
            return self.action
        if role is ComponentRole.OBJECT:
            return self.object_
        if role is ComponentRole.OUTCOME:
            return self.outcome
        if role is ComponentRole.TIME:
            return self.time
        raise ValueError(f"unknown role: {role}")


class ProbeType(str, Enum):
    """The four probe types in the MnemBound taxonomy (paper §9)."""

    SUPPORTED = "supported"
    ABSENT = "absent"
    PARTIAL = "partial"
    CROSS_BOUNDARY = "cross_boundary"


@dataclass(frozen=True)
class Probe:
    """A single probe with construction-time ground-truth labels."""

    probe_id: str
    probe_type: ProbeType
    recall: Recall
    fragments: tuple[Fragment, ...]

    is_fragment_supported: bool
    is_event_supported: bool

    boundary_event_pairs: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    # Bridge entities shared across the supporting fragments' source events.
    # Empty for SUPPORTED/ABSENT/PARTIAL probes. For CROSS_BOUNDARY traps,
    # populated when the probe was generated with prefer_bridge_linked=True
    # AND the trap's source events shared a bridge entity. Used for realism
    # scoring (§5 of the paper): bridge-linked traps are SOC-plausible
    # confusion patterns, while pure-random traps are formal-only.
    bridge_context: frozenset[str] = field(default_factory=frozenset)

    def is_mnemonic_boundary_violation(self) -> bool:
        return self.is_fragment_supported and not self.is_event_supported

    def is_bridge_linked(self) -> bool:
        """True iff this probe carries shared-bridge evidence across its source events."""
        return len(self.bridge_context) > 0


@dataclass(frozen=True)
class Corpus:
    """A complete MnemBound corpus."""

    seed: int
    salt: str
    records: tuple[Record, ...]
    fragments: tuple[Fragment, ...]
    probes: tuple[Probe, ...]

    tenant_entity_pools: Mapping[str, frozenset[str]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    # The entity sanitizer: { original_entity -> sanitized_token }. Used by
    # the Fragment dataclass when producing tenant-hidden views. Carried on
    # the corpus for inspection and so audit-trail replay can recover the
    # sanitized forms reproducibly.
    entity_sanitizer: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )

    # Store-level provenance resolver. Granted to BCESV; never V_frag.
    resolver: ProvenanceResolver | None = None

    # The BridgePools whitelist used during generation. Carried on the
    # corpus so validation, scoring, and realism audits can re-derive
    # which entities are cross-tenant bridges vs per-tenant entities.
    # Default is an empty BridgePools (no bridges configured).
    bridge_pools: object = None  # actually BridgePools; using object to avoid circular import

    def tenants(self) -> tuple[str, ...]:
        return tuple(sorted({r.client_id for r in self.records}))

    def records_for_tenant(self, tenant_id: str) -> tuple[Record, ...]:
        return tuple(r for r in self.records if r.client_id == tenant_id)
