"""
test_bcesv.py — smoke test for the BCESV verifier logic.

Builds a tiny synthetic dataset (5 fragments across 3 boundaries) and
checks that each verifier variant produces the expected verdicts.

Run with:  python3 test_bcesv.py
"""
from bcesv import (
    StoreEntry, FragmentOnly, BcesvExact, BcesvNoisy, BcesvReconstruct,
    ground_truth_boundary_invalid, all_supplying,
)


def make_store():
    """Three fragments from one event in boundary b-A, two from a
    different event in boundary b-B."""
    return {
        "prov:A1": StoreEntry(
            role="subject", value="user:alice@A", sanitized_value="user:u_a",
            text="Authentication by user:alice@A", sanitized_text="Authentication by user:u_a",
            public_provenance_id="prov:A1",
            record_id="r-A-1", boundary_id="b-A", event_id="e-A-1",
            tenant_id="tenant-A",
            timestamp="2026-03-10T10:00:00+00:00",
            log_source="edr:Crowdstrike", incident_template="auth_v1",
            entity_set=("user:alice@A", "host:wsA-01")),
        "prov:A2": StoreEntry(
            role="action", value="logged_in", sanitized_value="logged_in",
            text="logged in successfully", sanitized_text="logged in successfully",
            public_provenance_id="prov:A2",
            record_id="r-A-1", boundary_id="b-A", event_id="e-A-1",
            tenant_id="tenant-A",
            timestamp="2026-03-10T10:00:05+00:00",
            log_source="edr:Crowdstrike", incident_template="auth_v1",
            entity_set=("user:alice@A", "host:wsA-01")),
        "prov:A3": StoreEntry(
            role="object", value="host:wsA-01", sanitized_value="host:wsA-01",
            text="on host:wsA-01", sanitized_text="on host:wsA-01",
            public_provenance_id="prov:A3",
            record_id="r-A-1", boundary_id="b-A", event_id="e-A-1",
            tenant_id="tenant-A",
            timestamp="2026-03-10T10:00:10+00:00",
            log_source="edr:Crowdstrike", incident_template="auth_v1",
            entity_set=("user:alice@A", "host:wsA-01")),
        "prov:B1": StoreEntry(
            role="action", value="downloaded_file", sanitized_value="downloaded_file",
            text="downloaded file from", sanitized_text="downloaded file from",
            public_provenance_id="prov:B1",
            record_id="r-B-1", boundary_id="b-B", event_id="e-B-1",
            tenant_id="tenant-B",
            timestamp="2026-04-01T15:30:00+00:00",
            log_source="ngfw:Palo", incident_template="dlp_v2",
            entity_set=("user:bob@B", "ip:10.0.0.42")),
        "prov:B2": StoreEntry(
            role="object", value="ip:10.0.0.42", sanitized_value="ip:10.0.0.42",
            text="ip:10.0.0.42", sanitized_text="ip:10.0.0.42",
            public_provenance_id="prov:B2",
            record_id="r-B-1", boundary_id="b-B", event_id="e-B-1",
            tenant_id="tenant-B",
            timestamp="2026-04-01T15:30:02+00:00",
            log_source="ngfw:Palo", incident_template="dlp_v2",
            entity_set=("user:bob@B", "ip:10.0.0.42")),
    }


def case_safe_recall():
    """Recall composed only from boundary b-A — should be SAFE."""
    return {
        "probe_id": "test-safe",
        "probe_type": "supported",
        "rung": "shared_vector_index",
        "abstained": False,
        "value_view": "tenant_hidden",
        "fragment_provenance_ids": ["prov:A1", "prov:A2", "prov:A3", "prov:B1", "prov:B2"],
        "recall": {
            "subject": "user:u_a",
            "action": "logged in successfully",
            "object": "host:wsA-01",
            "outcome": "",
            "time": "",
        },
    }


def case_invalid_recall():
    """Recall composes across b-A and b-B — should be INVALID."""
    return {
        "probe_id": "test-invalid",
        "probe_type": "cross_boundary",
        "rung": "shared_vector_index",
        "abstained": False,
        "value_view": "tenant_hidden",
        "fragment_provenance_ids": ["prov:A1", "prov:A2", "prov:A3", "prov:B1", "prov:B2"],
        "recall": {
            "subject": "user:u_a",       # from b-A
            "action": "downloaded_file",  # from b-B
            "object": "ip:10.0.0.42",     # from b-B
            "outcome": "",
            "time": "",
        },
    }


def case_abstained():
    return {
        "probe_id": "test-abstain",
        "probe_type": "absent",
        "rung": "shared_vector_index",
        "abstained": True,
        "value_view": "tenant_hidden",
        "fragment_provenance_ids": ["prov:A1"],
        "recall": None,
    }


def run():
    store = make_store()
    cases = {
        "safe (single-boundary)": case_safe_recall(),
        "invalid (cross-boundary)": case_invalid_recall(),
        "abstained": case_abstained(),
    }

    print()
    print(f"{'verifier':28s} | {'safe':12s} | {'invalid':12s} | {'abstained':12s}")
    print("-" * 80)
    verifiers = [
        FragmentOnly(),
        BcesvExact(),
        BcesvNoisy(0.0, seed=1),
        BcesvNoisy(0.3, seed=1),
        BcesvReconstruct(),
    ]
    expected = {
        "FragmentOnly":         ("INVALID", "INVALID", "ABSTAIN"),
        "BcesvExact":           ("SAFE",    "INVALID", "ABSTAIN"),
        "BcesvNoisy(0%)":       ("SAFE",    "INVALID", "ABSTAIN"),
        "BcesvNoisy(30%)":      ("ANY",     "ANY",     "ABSTAIN"),
        "BcesvReconstruct":     ("SAFE",    "INVALID", "ABSTAIN"),
    }
    fails = 0
    for v in verifiers:
        results = [v.predict(c, store) for c in cases.values()]
        print(f"{v.name:28s} | "
              f"{results[0]:12s} | {results[1]:12s} | {results[2]:12s}")
        exp = expected.get(v.name)
        if exp:
            for got, want in zip(results, exp):
                if want != "ANY" and got != want:
                    print(f"  FAIL: {v.name}: expected {want}, got {got}")
                    fails += 1

    print()
    print(f"ground truth (boundary-invalid):")
    for label, c in cases.items():
        gt = ground_truth_boundary_invalid(c, store)
        print(f"  {label}: gt_invalid={gt}")

    print()
    if fails == 0:
        print("✓ All expected verdicts match.")
    else:
        print(f"✗ {fails} mismatches.")


if __name__ == "__main__":
    run()
