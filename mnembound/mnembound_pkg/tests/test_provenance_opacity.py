"""
Provenance and view opacity tests.

Verifies the §4.2 abstraction at the implementation level:

- view_for_v_frag(tenant_visible=True) exposes recall components and the
  opaque public provenance ID, with original entity strings. NO record_id,
  boundary_id, event_id, tenant_id, log_source, or timestamp keys.

- view_for_v_frag(tenant_visible=False) additionally strips tenant identity
  from value and text by substituting sanitizer tokens.

- Opaque public provenance IDs do not leak record/tenant grouping by their
  string structure.

- The ProvenanceResolver correctly maps opaque IDs back, is corpus-specific,
  and is the only path to store-level metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnembound.generator import generate_corpus
from mnembound.schema import (
    ALL_ROLES,
    PUBLIC_HANDLE_HEX_LEN,
    Fragment,
    ProvenanceHandle,
)


# Forbidden substrings in any V_frag-visible string in either view mode:
# these reveal the store-level record/boundary/event grouping that V_frag
# must not see.
def _make_grouping_forbidden_substrings(corpus) -> set[str]:
    forbidden: set[str] = set()
    for r in corpus.records:
        forbidden.add(r.record_id)
        forbidden.add(r.boundary_id)
        forbidden.add(r.event_id)
    return forbidden


# Tenant-revealing substrings: forbidden in tenant-hidden view only.
def _make_tenant_revealing_substrings(corpus) -> set[str]:
    forbidden = set()
    for tid in corpus.tenants():
        prefix_letter = tid.removeprefix("client")
        forbidden.add(tid)
        forbidden.add(f"-{prefix_letter}.")
        forbidden.add(f"-{prefix_letter}@")
        forbidden.add(f"/{prefix_letter}/")
        forbidden.add(f"_{prefix_letter}")
    return forbidden


def test_view_exposes_no_hidden_keys_in_either_mode() -> None:
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=50)
    allowed_keys = {"role", "value", "text", "provenance"}

    for f in corpus.fragments:
        for tenant_visible in (True, False):
            view = f.view_for_v_frag(tenant_visible=tenant_visible)
            extra = set(view.keys()) - allowed_keys
            assert not extra, f"view (tv={tenant_visible}) has extra keys: {extra}"
            missing = allowed_keys - set(view.keys())
            assert not missing, f"view (tv={tenant_visible}) missing keys: {missing}"

    print(f"[OK] every fragment view in both modes has exactly {sorted(allowed_keys)}")


def test_view_provenance_is_opaque_in_either_mode() -> None:
    """
    The provenance field is the opaque public ID in both modes. It does not
    leak record/boundary/event IDs.
    """
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=50)
    forbidden = _make_grouping_forbidden_substrings(corpus)

    for f in corpus.fragments:
        for tenant_visible in (True, False):
            view = f.view_for_v_frag(tenant_visible=tenant_visible)
            prov = view["provenance"]
            assert prov.startswith("prov:"), f"malformed opaque ID: {prov!r}"
            hex_part = prov[len("prov:"):]
            assert len(hex_part) == PUBLIC_HANDLE_HEX_LEN, (
                f"hex tail is {len(hex_part)} chars, expected "
                f"{PUBLIC_HANDLE_HEX_LEN}"
            )
            assert all(c in "0123456789abcdef" for c in hex_part)
            for bad in forbidden:
                assert bad not in prov, (
                    f"provenance {prov!r} leaks {bad!r}"
                )

    print(
        f"[OK] {len(corpus.fragments)} provenance IDs are {PUBLIC_HANDLE_HEX_LEN}-hex "
        "and leak no grouping info"
    )


def test_tenant_hidden_view_strips_tenant_identity() -> None:
    """
    In tenant-hidden mode, neither value nor text contains any tenant-
    revealing substring.
    """
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=50)
    forbidden = _make_tenant_revealing_substrings(corpus)

    for f in corpus.fragments:
        view = f.view_for_v_frag(tenant_visible=False)
        for bad in forbidden:
            assert bad not in view["value"], (
                f"tenant-hidden value {view['value']!r} leaks {bad!r}"
            )
            assert bad not in view["text"], (
                f"tenant-hidden text {view['text']!r} leaks {bad!r}"
            )

    print("[OK] tenant-hidden view strips tenant-revealing substrings from value and text")


def test_tenant_visible_view_intentionally_contains_tenant_strings() -> None:
    """
    In tenant-visible mode, entity strings include tenant prefixes (this is
    the whole point of the visible view; §4.4 measures this baseline
    explicitly). Verify a sample shows the original entity is preserved.
    """
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=50)
    record_index = {r.record_id: r for r in corpus.records}

    seen_tenant_visible = False
    for f in corpus.fragments:
        view = f.view_for_v_frag(tenant_visible=True)
        r = record_index[f.record_id]
        # The role's component value should appear verbatim in the visible
        # view.
        assert view["value"] == r.component(f.role)
        # If the value has a tenant prefix in it (subjects and objects do),
        # confirm it's still there.
        prefix_letter = r.client_id.removeprefix("client")
        if (
            f"-{prefix_letter}." in r.subject
            or f"-{prefix_letter}@" in r.subject
            or f"/{prefix_letter}/" in r.subject
        ):
            seen_tenant_visible = True

    assert seen_tenant_visible, (
        "no fragment exposed a tenant-bearing entity in visible mode; "
        "test setup may be wrong"
    )
    print("[OK] tenant-visible view preserves original tenant-revealing entity strings")


def test_view_text_does_not_leak_grouping_ids() -> None:
    """
    Fragment text must not contain record_id or event_id strings (these have
    unique 6-digit numeric suffixes and cannot appear in any naturalistic text
    by chance).

    For boundary_id under TENANT granularity, the form is 'b-A', 'b-B', etc.
    — short two-character tails that can coincidentally appear in normal
    English (e.g., 'bob-A' contains 'b-A'). The relevant check for boundary
    leakage is whether the full boundary token, when bounded by whitespace
    or a leading dash, appears in text. We check the punctuation-bounded
    form, which corresponds to what a verifier could exploit.
    """
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=50)
    for f in corpus.fragments:
        for tenant_visible in (True, False):
            view = f.view_for_v_frag(tenant_visible=tenant_visible)
            text = view["text"]
            assert f.record_id not in text, f"text leaks record_id {f.record_id}"
            assert f.event_id not in text, f"text leaks event_id {f.event_id}"
            # Check boundary_id only as a punctuation-bounded token.
            # 'b-A' may appear as a substring of 'bob-A' but that is not
            # boundary identity leakage; a verifier without schema knowledge
            # cannot recover that 'b-A' was a boundary identifier.
            boundary_as_token = f" {f.boundary_id} "
            assert boundary_as_token not in f" {text} ", (
                f"text leaks boundary_id token {f.boundary_id} in {text!r}"
            )

    print("[OK] no fragment text in either mode leaks record/event IDs or boundary tokens")


def test_fragments_from_same_record_have_unrelated_public_ids() -> None:
    """
    Public IDs of fragments from one record share only the 'prov:' marker
    structurally. Random hex collisions are expected at chance rate.
    """
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=50)

    frags_by_record: dict[str, list[Fragment]] = {}
    for f in corpus.fragments:
        frags_by_record.setdefault(f.record_id, []).append(f)

    PROV_MARKER = "prov:"
    common_prefix_lengths: list[int] = []

    for rid, frags in frags_by_record.items():
        public_ids = [f.public_provenance_id for f in frags]
        assert len(public_ids) == 5
        for i in range(len(public_ids)):
            for j in range(i + 1, len(public_ids)):
                a, b = public_ids[i], public_ids[j]
                assert a.startswith(PROV_MARKER) and b.startswith(PROV_MARKER)
                a_tail, b_tail = a[len(PROV_MARKER):], b[len(PROV_MARKER):]
                k = 0
                while k < min(len(a_tail), len(b_tail)) and a_tail[k] == b_tail[k]:
                    k += 1
                common_prefix_lengths.append(k)
                # 5-char LCP from random hex strings has p ~= 9.5e-7. A
                # structural leak would show length ~14+.
                assert k <= 5, (
                    f"record {rid} public IDs {a} and {b} share {k} hex chars "
                    "of tail prefix; structural opacity leak suspected"
                )

    mean_lcp = sum(common_prefix_lengths) / len(common_prefix_lengths)
    assert mean_lcp < 1.0, (
        f"mean within-record common prefix {mean_lcp:.3f} is unexpectedly high"
    )
    print(
        f"[OK] within-record public IDs show random-baseline prefix sharing "
        f"(mean LCP beyond 'prov:' = {mean_lcp:.3f} chars)"
    )


def test_fragments_from_same_tenant_have_unrelated_public_ids() -> None:
    """Stronger check across many records within one tenant."""
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=50)

    frags_by_tenant: dict[str, list[Fragment]] = {}
    for f in corpus.fragments:
        frags_by_tenant.setdefault(f.tenant_id, []).append(f)

    PROV_MARKER = "prov:"
    overall_lcps: list[int] = []

    for tid, frags in frags_by_tenant.items():
        public_ids = sorted({f.public_provenance_id for f in frags})[:50]
        for i in range(len(public_ids)):
            for j in range(i + 1, len(public_ids)):
                a, b = public_ids[i], public_ids[j]
                a_tail, b_tail = a[len(PROV_MARKER):], b[len(PROV_MARKER):]
                k = 0
                while k < min(len(a_tail), len(b_tail)) and a_tail[k] == b_tail[k]:
                    k += 1
                overall_lcps.append(k)
                assert k <= 5, (
                    f"tenant {tid} public IDs {a} and {b} share {k} chars"
                )

    mean_lcp = sum(overall_lcps) / len(overall_lcps)
    assert mean_lcp < 1.0
    print(
        f"[OK] within-tenant public IDs show no tenant-level prefix structure "
        f"(mean LCP = {mean_lcp:.3f} chars across {len(overall_lcps)} pairs)"
    )


def test_resolver_resolves_correctly() -> None:
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=50)
    resolver = corpus.resolver
    assert resolver is not None

    for f in corpus.fragments:
        rid, fidx = resolver.resolve(f.public_provenance_id)
        assert rid == f.record_id
        assert fidx == f.provenance.fragment_index

    try:
        resolver.resolve("prov:" + "0" * PUBLIC_HANDLE_HEX_LEN)
        raise AssertionError("resolver should raise on unknown public_id")
    except KeyError:
        pass

    print(f"[OK] resolver correctly maps {len(corpus.fragments)} public IDs")


def test_resolver_is_corpus_specific() -> None:
    """Different seeds -> different salts -> resolvers do not cross corpora."""
    cA = generate_corpus(seed=1, n_tenants=5, records_per_tenant=20)
    cB = generate_corpus(seed=2, n_tenants=5, records_per_tenant=20)
    assert cA.salt != cB.salt

    try:
        cA.resolver.resolve(cB.fragments[0].public_provenance_id)
        raise AssertionError("corpus A resolver resolved a corpus B id")
    except KeyError:
        pass
    print("[OK] resolvers are corpus-specific (salt-isolated)")


def test_public_ids_are_globally_unique_within_corpus() -> None:
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=100)
    ids = [f.public_provenance_id for f in corpus.fragments]
    assert len(ids) == len(set(ids))
    print(f"[OK] all {len(ids)} public_provenance_ids are unique within corpus")


def test_sanitizer_is_injective() -> None:
    """Different entities map to different sanitizer tokens."""
    corpus = generate_corpus(seed=42, n_tenants=10, records_per_tenant=50)
    sanitizer = corpus.entity_sanitizer
    assert len(sanitizer) > 0
    tokens = set(sanitizer.values())
    assert len(tokens) == len(sanitizer), (
        f"sanitizer has {len(sanitizer)} entries but only {len(tokens)} "
        "unique tokens; collision"
    )
    print(f"[OK] sanitizer maps {len(sanitizer)} entities to unique tokens")


def main() -> int:
    tests = [
        test_view_exposes_no_hidden_keys_in_either_mode,
        test_view_provenance_is_opaque_in_either_mode,
        test_tenant_hidden_view_strips_tenant_identity,
        test_tenant_visible_view_intentionally_contains_tenant_strings,
        test_view_text_does_not_leak_grouping_ids,
        test_fragments_from_same_record_have_unrelated_public_ids,
        test_fragments_from_same_tenant_have_unrelated_public_ids,
        test_resolver_resolves_correctly,
        test_resolver_is_corpus_specific,
        test_public_ids_are_globally_unique_within_corpus,
        test_sanitizer_is_injective,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failures.append((t.__name__, e))
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"=== {len(failures)} test(s) failed ===")
        return 1
    print(f"=== all {len(tests)} opacity tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
