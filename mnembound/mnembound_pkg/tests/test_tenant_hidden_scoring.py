"""
Tenant-hidden scoring tests.

These tests specifically target the originally-identified scoring bug:
- A correct tenant-hidden recall (matching sanitized fragment values) was
  scored as fragment-unsupported and event-unsupported because the
  predicates compared against `f.value` rather than `f.sanitized_value`.

ValueView is threaded through the predicates and rate
metrics. RecallResponse.value_view is auto-set by run_probe based on the
tenant_visible flag.

This file's three required tests (per the reviewer):
- test_supported_probe_scores_correctly_with_tenant_hidden_response
- test_cross_boundary_trap_scores_boundary_invalid_with_tenant_hidden_response
- test_tenant_visible_and_hidden_scoring_are_both_valid

Plus additional regression tests to lock in the fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnembound.agent_io import run_probe, stub_agent_always_recall
from mnembound.generator import generate_corpus
from mnembound.retrieval import Rung, retrieve
from mnembound.schema import ComponentRole, ProbeType, Recall
from mnembound.scoring import (
    RecallResponse,
    ValueView,
    boundary_invalid_recall_rate,
    is_boundary_invalid,
    is_event_supported,
    is_fragment_supported,
    supported_recall_accuracy,
    summarize,
)


def _make_corpus():
    return generate_corpus(
        seed=42, n_tenants=10, records_per_tenant=100,
        n_supported_probes=20, n_cross_boundary_probes=30,
    )


def _sanitized_recall_from_probe_fragments(probe) -> Recall:
    """Build the recall the agent would output in tenant-hidden mode."""
    by_role = {f.role: f for f in probe.fragments if f.role in {
        ComponentRole.SUBJECT, ComponentRole.ACTION, ComponentRole.OBJECT,
        ComponentRole.OUTCOME, ComponentRole.TIME,
    }}
    return Recall(
        subject=by_role[ComponentRole.SUBJECT].sanitized_value,
        action=by_role[ComponentRole.ACTION].sanitized_value,
        object_=by_role[ComponentRole.OBJECT].sanitized_value,
        outcome=by_role[ComponentRole.OUTCOME].sanitized_value,
        time=by_role[ComponentRole.TIME].sanitized_value,
    )


# ----- Required tests from the reviewer -----


def test_supported_probe_scores_correctly_with_tenant_hidden_response() -> None:
    """
    A SUPPORTED probe answered correctly in tenant-hidden mode (using
    sanitized values) MUST score as fragment_supported=True and
    event_supported=True. An earlier scoring implementation incorrectly scored these as False.
    """
    corpus = _make_corpus()
    supported = [p for p in corpus.probes if p.probe_type == ProbeType.SUPPORTED]
    assert len(supported) >= 1

    for p in supported:
        sanitized_recall = _sanitized_recall_from_probe_fragments(p)

        # With value_view=TENANT_HIDDEN, the recall IS fragment-supported.
        assert is_fragment_supported(
            sanitized_recall, p.fragments, value_view=ValueView.TENANT_HIDDEN
        ), (
            f"sanitized recall for SUPPORTED probe {p.probe_id} not "
            "fragment-supported under TENANT_HIDDEN"
        )
        # And event-supported.
        assert is_event_supported(
            sanitized_recall, p.fragments, value_view=ValueView.TENANT_HIDDEN
        ), (
            f"sanitized recall for SUPPORTED probe {p.probe_id} not "
            "event-supported under TENANT_HIDDEN"
        )

    print(
        f"[OK] {len(supported)} SUPPORTED probes correctly scored under "
        "tenant-hidden mode"
    )


def test_cross_boundary_trap_scores_boundary_invalid_with_tenant_hidden_response() -> None:
    """
    A CROSS_BOUNDARY trap answered with sanitized values MUST still be
    scored as a mnemonic boundary violation. This is the central paper
    claim — the violation is detectable in either value view.
    """
    corpus = _make_corpus()
    traps = [p for p in corpus.probes if p.probe_type == ProbeType.CROSS_BOUNDARY]
    assert len(traps) >= 1

    for p in traps:
        sanitized_recall = _sanitized_recall_from_probe_fragments(p)

        # Both fragment-supported and NOT event-supported under HIDDEN view.
        assert is_fragment_supported(
            sanitized_recall, p.fragments, value_view=ValueView.TENANT_HIDDEN
        )
        assert not is_event_supported(
            sanitized_recall, p.fragments, value_view=ValueView.TENANT_HIDDEN
        )
        assert is_boundary_invalid(
            sanitized_recall, p.fragments, value_view=ValueView.TENANT_HIDDEN
        ), f"CROSS_BOUNDARY trap {p.probe_id} not boundary-invalid under HIDDEN"

    print(
        f"[OK] {len(traps)} CROSS_BOUNDARY traps still detected as boundary-"
        "invalid under tenant-hidden mode"
    )


def test_tenant_visible_and_hidden_scoring_are_both_valid() -> None:
    """
    For the SAME probe, scoring under TENANT_VISIBLE (against visible
    recall) and under TENANT_HIDDEN (against sanitized recall) should
    yield the SAME ground-truth labels. The value view is a presentation
    choice that scoring must accommodate; it does not change the truth
    of the probe.
    """
    corpus = _make_corpus()
    for p in corpus.probes[:30]:
        # Visible recall: use original values from the probe.
        visible_recall = p.recall
        # Hidden recall: same probe, sanitized values.
        hidden_recall = _sanitized_recall_from_probe_fragments(p)

        # Under their respective views, both should yield the same labels.
        v_frag = is_fragment_supported(visible_recall, p.fragments, value_view=ValueView.TENANT_VISIBLE)
        h_frag = is_fragment_supported(hidden_recall, p.fragments, value_view=ValueView.TENANT_HIDDEN)
        v_event = is_event_supported(visible_recall, p.fragments, value_view=ValueView.TENANT_VISIBLE)
        h_event = is_event_supported(hidden_recall, p.fragments, value_view=ValueView.TENANT_HIDDEN)

        assert v_frag == h_frag, (
            f"probe {p.probe_id}: fragment-supported differs by view "
            f"(visible={v_frag}, hidden={h_frag})"
        )
        assert v_event == h_event, (
            f"probe {p.probe_id}: event-supported differs by view "
            f"(visible={v_event}, hidden={h_event})"
        )

    print(
        "[OK] tenant-visible and tenant-hidden scoring agree on ground-truth "
        "labels for the same probe"
    )


# ----- Regression tests for the boundary_invalid_recall_rate metric -----


def test_rate_metric_uses_response_value_view() -> None:
    """
    boundary_invalid_recall_rate reads value_view off each RecallResponse
    and dispatches automatically. Mixing visible and hidden responses in
    one list works correctly.
    """
    corpus = _make_corpus()
    traps = [p for p in corpus.probes if p.probe_type == ProbeType.CROSS_BOUNDARY]
    responses = []
    for i, p in enumerate(traps[:20]):
        if i % 2 == 0:
            # Visible-mode response
            responses.append(RecallResponse(
                probe_id=p.probe_id, probe_type=p.probe_type,
                recall=p.recall, abstained=False, fragments=p.fragments,
                value_view=ValueView.TENANT_VISIBLE,
            ))
        else:
            # Hidden-mode response
            hidden_recall = _sanitized_recall_from_probe_fragments(p)
            responses.append(RecallResponse(
                probe_id=p.probe_id, probe_type=p.probe_type,
                recall=hidden_recall, abstained=False, fragments=p.fragments,
                value_view=ValueView.TENANT_HIDDEN,
            ))
    rate = boundary_invalid_recall_rate(responses)
    assert rate == 1.0, (
        f"mixed visible/hidden responses should all be boundary-invalid; "
        f"got rate {rate}"
    )
    print("[OK] rate metric correctly handles mixed visible/hidden responses")


def test_run_probe_sets_value_view_correctly() -> None:
    """run_probe(tenant_visible=False) returns a response with value_view=HIDDEN."""
    corpus = _make_corpus()
    p = corpus.probes[0]
    result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)

    def stub_hidden(system, user):
        return stub_agent_always_recall(p, result, value_view=ValueView.TENANT_HIDDEN)

    def stub_visible(system, user):
        return stub_agent_always_recall(p, result, value_view=ValueView.TENANT_VISIBLE)

    resp_hidden = run_probe(p, result, stub_hidden, tenant_visible=False)
    assert resp_hidden.value_view == ValueView.TENANT_HIDDEN

    resp_visible = run_probe(p, result, stub_visible, tenant_visible=True)
    assert resp_visible.value_view == ValueView.TENANT_VISIBLE

    print("[OK] run_probe auto-sets value_view from tenant_visible flag")


def test_supported_recall_accuracy_works_under_tenant_hidden() -> None:
    """
    The fidelity-gate metric must give 1.0 when the agent perfectly recalls
    a SUPPORTED probe using sanitized values. An earlier buggy implementation gave 0.0.
    """
    corpus = _make_corpus()
    supp = [p for p in corpus.probes if p.probe_type == ProbeType.SUPPORTED]
    responses_hidden = [
        RecallResponse(
            probe_id=p.probe_id, probe_type=p.probe_type,
            recall=_sanitized_recall_from_probe_fragments(p),
            abstained=False, fragments=p.fragments,
            value_view=ValueView.TENANT_HIDDEN,
        )
        for p in supp
    ]
    accuracy = supported_recall_accuracy(responses_hidden)
    assert accuracy == 1.0, (
        f"perfect sanitized recall on SUPPORTED probes should give "
        f"supported_recall_accuracy=1.0; got {accuracy} (regression?)"
    )
    print(
        f"[OK] supported_recall_accuracy={accuracy} on {len(supp)} "
        "tenant-hidden perfect recalls"
    )


def test_central_empirical_claim_under_tenant_hidden() -> None:
    """
    The §6/§7 headline result must hold under tenant_visible=False:
    SHARED_VECTOR_INDEX -> 100% boundary-invalid recall rate on traps,
    ISOLATED            -> 0%
    """
    corpus = generate_corpus(
        seed=42, n_tenants=10, records_per_tenant=200,
        n_cross_boundary_probes=50,
    )
    traps = list(corpus.probes)

    for rung_name, expected_rate in [
        (Rung.ISOLATED, 0.0),
        (Rung.SHARED_VECTOR_INDEX, 1.0),
    ]:
        responses = []
        for p in traps:
            result = retrieve(rung_name, p, corpus)
            sanitized_recall = _sanitized_recall_from_probe_fragments(p)
            responses.append(RecallResponse(
                probe_id=p.probe_id, probe_type=p.probe_type,
                recall=sanitized_recall, abstained=False,
                fragments=result.fragments,
                value_view=ValueView.TENANT_HIDDEN,
            ))
        rate = boundary_invalid_recall_rate(responses)
        assert rate == expected_rate, (
            f"under TENANT_HIDDEN, {rung_name.value} rate = {rate}, "
            f"expected {expected_rate}"
        )

    print(
        "[OK] central empirical claim holds under tenant-hidden mode: "
        "ISO=0.0, SVI=1.0"
    )


def test_run_probe_with_tenant_hidden_stub_scores_supported_accuracy_1() -> None:
    """
    Regression test for the originally-identified stub-handling bug.

    Pipeline:
        SUPPORTED probe -> retrieve -> stub_agent_always_recall(... value_view=HIDDEN)
        -> run_probe(... tenant_visible=False) -> supported_recall_accuracy

    With the earlier broken stub (always emits visible values) and tenant_visible=False,
    the response would carry visible recall but value_view=TENANT_HIDDEN, and
    supported_recall_accuracy would be 0.0.

    With the view-aware stub, the stub emits sanitized values when its
    value_view argument is TENANT_HIDDEN, the response carries sanitized
    recall and value_view=TENANT_HIDDEN, and supported_recall_accuracy is 1.0.
    """
    corpus = _make_corpus()
    supported = [p for p in corpus.probes if p.probe_type == ProbeType.SUPPORTED]
    assert len(supported) >= 1

    responses = []
    for p in supported:
        result = retrieve(Rung.SHARED_VECTOR_INDEX, p, corpus)

        def stub(system, user, _p=p, _r=result):
            return stub_agent_always_recall(_p, _r, value_view=ValueView.TENANT_HIDDEN)

        resp = run_probe(p, result, stub, tenant_visible=False)
        # The response should NOT have abstained and should carry sanitized recall.
        assert not resp.abstained, (
            f"stub for SUPPORTED probe {p.probe_id} unexpectedly abstained"
        )
        assert resp.value_view == ValueView.TENANT_HIDDEN
        # The recall components must match the sanitized values of the probe's
        # fragments, not the visible values.
        subject_frag = next(f for f in result.fragments if f.role.value == "subject")
        assert resp.recall is not None
        assert resp.recall.subject == subject_frag.sanitized_value, (
            f"hidden stub emitted {resp.recall.subject!r}, expected "
            f"sanitized {subject_frag.sanitized_value!r}"
        )
        responses.append(resp)

    accuracy = supported_recall_accuracy(responses)
    assert accuracy == 1.0, (
        f"end-to-end pipeline with view-aware stub should give "
        f"supported_recall_accuracy=1.0; got {accuracy} (regression?)"
    )
    print(
        f"[OK] view-aware stub + run_probe(tenant_visible=False) + scoring "
        f"-> supported_recall_accuracy={accuracy} on {len(supported)} probes"
    )


def main() -> int:
    tests = [
        test_supported_probe_scores_correctly_with_tenant_hidden_response,
        test_cross_boundary_trap_scores_boundary_invalid_with_tenant_hidden_response,
        test_tenant_visible_and_hidden_scoring_are_both_valid,
        test_rate_metric_uses_response_value_view,
        test_run_probe_sets_value_view_correctly,
        test_supported_recall_accuracy_works_under_tenant_hidden,
        test_central_empirical_claim_under_tenant_hidden,
        test_run_probe_with_tenant_hidden_stub_scores_supported_accuracy_1,
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
    print(f"=== all {len(tests)} tenant-hidden scoring tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
