"""
Export-safety tests.

Mechanically verify that the export functions in `mnembound.export` do
not accidentally leak store-level metadata into the FragmentViewBundle
intended for V_frag baselines.

The §4.2 abstraction holds at the IMPLEMENTATION level only if no
fragment-view dict contains any forbidden key (record_id, boundary_id,
event_id, tenant_id, salt, resolver, etc.). If a future refactor breaks
this, these tests catch it before the artifact ships.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnembound.export import (
    FORBIDDEN_KEYS_IN_FRAGMENT_VIEW,
    assert_fragment_view_is_clean,
    export_bcesv_view,
    export_fragment_view,
    export_store_view,
)
from mnembound.generator import generate_corpus


def _make_corpus():
    return generate_corpus(seed=42, n_tenants=10, records_per_tenant=50)


def test_fragment_view_export_has_only_allowed_keys() -> None:
    """
    Each fragment-view dict has exactly {role, value, text, provenance}
    and NO forbidden keys.
    """
    corpus = _make_corpus()
    for tv in (True, False):
        bundle = export_fragment_view(corpus, tenant_visible=tv)
        allowed = {"role", "value", "text", "provenance"}
        for i, view in enumerate(bundle.fragments):
            keys = set(view.keys())
            assert keys == allowed, (
                f"fragment view {i} (tv={tv}) has keys {sorted(keys)}, "
                f"expected exactly {sorted(allowed)}"
            )

    print("[OK] FragmentViewBundle has exactly {role, value, text, provenance}")


def test_fragment_view_export_contains_no_forbidden_keys() -> None:
    """Strict redundant check via assert_fragment_view_is_clean."""
    corpus = _make_corpus()
    for tv in (True, False):
        bundle = export_fragment_view(corpus, tenant_visible=tv)
        assert_fragment_view_is_clean(bundle)
    print(
        f"[OK] FragmentViewBundle contains no forbidden keys "
        f"({sorted(FORBIDDEN_KEYS_IN_FRAGMENT_VIEW)})"
    )


def test_fragment_view_export_does_not_carry_resolver_or_salt() -> None:
    """The bundle ITSELF must not carry the resolver, sanitizer, or salt."""
    corpus = _make_corpus()
    bundle = export_fragment_view(corpus, tenant_visible=False)
    bundle_attrs = set(vars(bundle).keys())
    # FragmentViewBundle dataclass attributes:
    assert bundle_attrs == {"tenant_visible", "fragments"}, (
        f"FragmentViewBundle has extra attributes: {bundle_attrs}"
    )
    print(f"[OK] FragmentViewBundle has only {{tenant_visible, fragments}}")


def test_fragment_view_export_requires_tenant_visible_keyword() -> None:
    """
    `export_fragment_view` requires tenant_visible to be passed as a
    keyword (no default), preventing accidental omission.
    """
    corpus = _make_corpus()
    try:
        export_fragment_view(corpus)
        raise AssertionError("export_fragment_view should require tenant_visible")
    except TypeError:
        pass
    print("[OK] export_fragment_view requires explicit tenant_visible")


def test_tenant_hidden_export_strips_tenant_identity() -> None:
    """
    When tenant_visible=False, no fragment view's value or text contains
    a tenant prefix like 'clientA' or '-A.'.
    """
    corpus = _make_corpus()
    bundle = export_fragment_view(corpus, tenant_visible=False)

    forbidden = set()
    for tid in corpus.tenants():
        prefix_letter = tid.removeprefix("client")
        forbidden.add(tid)
        forbidden.add(f"-{prefix_letter}.")
        forbidden.add(f"-{prefix_letter}@")
        forbidden.add(f"/{prefix_letter}/")
        forbidden.add(f"_{prefix_letter}")

    for view in bundle.fragments:
        for bad in forbidden:
            assert bad not in view["value"], (
                f"tenant-hidden export leaks {bad!r} in value {view['value']!r}"
            )
            assert bad not in view["text"], (
                f"tenant-hidden export leaks {bad!r} in text {view['text']!r}"
            )

    print("[OK] tenant-hidden export strips tenant identity from value and text")


def test_bcesv_export_carries_store_metadata() -> None:
    """BCESV export gives full per-fragment metadata + the resolver."""
    corpus = _make_corpus()
    bundle = export_bcesv_view(corpus)

    assert bundle.resolver is corpus.resolver
    assert len(bundle.fragments) == len(corpus.fragments)
    assert len(bundle.fragments_by_public_id) == len(corpus.fragments)

    # Every BCESV fragment record carries the hidden metadata BCESV needs.
    for bf in bundle.fragments:
        assert bf.boundary_id
        assert bf.event_id
        assert bf.tenant_id
        assert bf.timestamp
        assert bf.log_source
        assert bf.incident_template
        assert len(bf.entity_set) >= 1

    print(f"[OK] BCESV export carries full metadata for {len(bundle.fragments)} fragments")


def test_bcesv_resolver_resolves_export_ids() -> None:
    """Public IDs from the BCESV bundle resolve via the bundle's resolver."""
    corpus = _make_corpus()
    bundle = export_bcesv_view(corpus)
    for bf in bundle.fragments[:50]:
        rid, fidx = bundle.resolver.resolve(bf.public_provenance_id)
        # Cross-check against the original fragment.
        original = next(
            f for f in corpus.fragments
            if f.public_provenance_id == bf.public_provenance_id
        )
        assert rid == original.record_id
        assert fidx == original.provenance.fragment_index
    print("[OK] BCESV bundle resolver correctly resolves all sampled public IDs")


def test_store_view_export_is_complete() -> None:
    """Store view contains the full corpus state used by scoring/validation."""
    corpus = _make_corpus()
    bundle = export_store_view(corpus)

    assert bundle.records == corpus.records
    assert bundle.fragments == corpus.fragments
    assert bundle.resolver is corpus.resolver
    assert dict(bundle.tenant_entity_pools) == dict(corpus.tenant_entity_pools)
    assert dict(bundle.entity_sanitizer) == dict(corpus.entity_sanitizer)
    assert bundle.salt == corpus.salt

    print("[OK] store-view export carries full corpus state")


def test_fragment_view_dicts_are_immutable() -> None:
    """
    Fragment-view dicts in the bundle are MappingProxyType views, so a
    consumer cannot mutate them and accidentally inject hidden keys.
    """
    corpus = _make_corpus()
    bundle = export_fragment_view(corpus, tenant_visible=False)
    sample = bundle.fragments[0]
    try:
        sample["record_id"] = "leak"  # type: ignore[index]
        raise AssertionError("mutation should have raised TypeError")
    except TypeError:
        pass
    print("[OK] fragment-view dicts in FragmentViewBundle are immutable")


def main() -> int:
    tests = [
        test_fragment_view_export_has_only_allowed_keys,
        test_fragment_view_export_contains_no_forbidden_keys,
        test_fragment_view_export_does_not_carry_resolver_or_salt,
        test_fragment_view_export_requires_tenant_visible_keyword,
        test_tenant_hidden_export_strips_tenant_identity,
        test_bcesv_export_carries_store_metadata,
        test_bcesv_resolver_resolves_export_ids,
        test_store_view_export_is_complete,
        test_fragment_view_dicts_are_immutable,
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
    print(f"=== all {len(tests)} export-safety tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
