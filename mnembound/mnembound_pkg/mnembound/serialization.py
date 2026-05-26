"""
mnembound.serialization — JSONL export and import for artifacts.

Each component of the experiment is written to its own JSONL file. The
canonical layout:

    corpus.json                   -- Metadata only (seed, salt, counts)
    records.jsonl                 -- One Record per line (no fragments)
    fragments.tenant_visible.jsonl
    fragments.tenant_hidden.jsonl -- View bundles for V_frag baselines
    fragments.store.jsonl         -- Full fragment data with hidden metadata
                                     (BCESV / scoring only; NEVER hand to a
                                     V_frag baseline)
    probes.jsonl                  -- One Probe per line with construction labels
    responses.jsonl               -- One RecallResponse per line
    metrics.json                  -- MetricSummary JSON
    rung_stats.jsonl              -- Per-rung RungSummary, one per line

Imports
-------
Each export has a corresponding load_* function so the artifact can be
re-hydrated for re-scoring or replay. Loading does NOT reconstruct the
ProvenanceResolver or entity sanitizer (those are corpus-internal); for
that, regenerate the corpus from its seed.

Privacy note
------------
Calling `write_fragments_tenant_hidden_jsonl` is the safe way to hand
fragment data to a V_frag baseline. The file contains ONLY
{role, value, text, provenance} per line — same as
FragmentViewBundle.fragments. The hidden-metadata `fragments.store.jsonl`
must NEVER be given to a V_frag baseline; it contains tenant_id,
boundary_id, event_id, etc.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from mnembound.export import (
    FORBIDDEN_KEYS_IN_FRAGMENT_VIEW,
    assert_fragment_view_is_clean,
    export_fragment_view,
)
from mnembound.retrieval import RungSummary
from mnembound.schema import (
    ALL_ROLES,
    ComponentRole,
    Corpus,
    Fragment,
    Probe,
    ProbeType,
    Recall,
    Record,
)
from mnembound.scoring import MetricSummary, RecallResponse


# ---------- Corpus + records ----------


def write_corpus_metadata(corpus: Corpus, path: str | Path) -> None:
    """
    Write corpus-level metadata: seed, salt, counts.

    Does NOT write records or fragments — those go in their own files.
    """
    data = {
        "seed": corpus.seed,
        "salt": corpus.salt,
        "n_records": len(corpus.records),
        "n_fragments": len(corpus.fragments),
        "n_probes": len(corpus.probes),
        "tenants": list(corpus.tenants()),
        "n_tenant_entity_pools": len(corpus.tenant_entity_pools),
        "n_sanitizer_entries": len(corpus.entity_sanitizer),
        "has_bridges": corpus.bridge_pools is not None and bool(
            getattr(corpus.bridge_pools, "shared_ips", None)
            or getattr(corpus.bridge_pools, "shared_hashes", None)
        ),
    }
    Path(path).write_text(json.dumps(data, indent=2))


def record_to_dict(record: Record) -> dict[str, Any]:
    """
    Convert a Record to a plain dict. The component_text is materialized
    as a {role_name -> text} dict for JSON friendliness; extra is converted
    to a plain dict.
    """
    return {
        "record_id": record.record_id,
        "client_id": record.client_id,
        "boundary_id": record.boundary_id,
        "event_id": record.event_id,
        "subject": record.subject,
        "action": record.action,
        "object": record.object_,
        "outcome": record.outcome,
        "time": record.time,
        "log_source": record.log_source,
        "incident_template": record.incident_template,
        "component_text": {
            role.value: record.text_for(role) for role in ALL_ROLES
        },
        "extra": dict(record.extra),
    }


def write_records_jsonl(records: Iterable[Record], path: str | Path) -> int:
    """Write one Record per line. Returns the number of lines written."""
    n = 0
    with Path(path).open("w") as fh:
        for r in records:
            fh.write(json.dumps(record_to_dict(r)))
            fh.write("\n")
            n += 1
    return n


# ---------- Fragments ----------


def write_fragments_tenant_visible_jsonl(
    corpus: Corpus, path: str | Path
) -> int:
    """
    Write per-fragment tenant-visible view (V_frag-safe).

    Each line is exactly {role, value, text, provenance}. No hidden
    metadata. This file is safe to hand to any V_frag baseline.
    """
    bundle = export_fragment_view(corpus, tenant_visible=True)
    assert_fragment_view_is_clean(bundle)
    n = 0
    with Path(path).open("w") as fh:
        for view in bundle.fragments:
            fh.write(json.dumps(dict(view)))
            fh.write("\n")
            n += 1
    return n


def write_fragments_tenant_hidden_jsonl(
    corpus: Corpus, path: str | Path
) -> int:
    """Same as visible, but with tenant-hidden value/text."""
    bundle = export_fragment_view(corpus, tenant_visible=False)
    assert_fragment_view_is_clean(bundle)
    n = 0
    with Path(path).open("w") as fh:
        for view in bundle.fragments:
            fh.write(json.dumps(dict(view)))
            fh.write("\n")
            n += 1
    return n


def fragment_to_store_dict(f: Fragment) -> dict[str, Any]:
    """
    Convert a Fragment to a dict including ALL fields.

    This is the store-level view: includes tenant_id, boundary_id,
    event_id, timestamp, log_source, etc. Suitable for scoring,
    validation, BCESV. NEVER expose this to a V_frag baseline.
    """
    return {
        "role": f.role.value,
        "value": f.value,
        "sanitized_value": f.sanitized_value,
        "text": f.text,
        "sanitized_text": f.sanitized_text,
        "public_provenance_id": f.public_provenance_id,
        "record_id": f.record_id,
        "boundary_id": f.boundary_id,
        "event_id": f.event_id,
        "tenant_id": f.tenant_id,
        "timestamp": f.timestamp,
        "log_source": f.log_source,
        "incident_template": f.incident_template,
        "entity_set": sorted(f.entity_set),
    }


def write_fragments_store_jsonl(corpus: Corpus, path: str | Path) -> int:
    """
    Write per-fragment FULL data (store-level). Used by scoring/validation.

    Privacy: this file contains tenant_id, boundary_id, event_id and
    other hidden metadata. Never give it to a V_frag baseline; that
    would invalidate the §7 audit-invisibility result.
    """
    n = 0
    with Path(path).open("w") as fh:
        for f in corpus.fragments:
            fh.write(json.dumps(fragment_to_store_dict(f)))
            fh.write("\n")
            n += 1
    return n


# ---------- Probes ----------


def recall_to_dict(recall: Recall) -> dict[str, str]:
    return {
        "subject": recall.subject,
        "action": recall.action,
        "object": recall.object_,
        "outcome": recall.outcome,
        "time": recall.time,
    }


def probe_to_dict(probe: Probe) -> dict[str, Any]:
    """
    Convert a Probe to a dict.

    Fragments are represented by their public_provenance_id (the V_frag
    handle); to reconstruct the full Fragment, look up by ID in the
    fragments file. This keeps probes.jsonl compact.
    """
    return {
        "probe_id": probe.probe_id,
        "probe_type": probe.probe_type.value,
        "recall": recall_to_dict(probe.recall),
        "fragment_provenance_ids": [
            f.public_provenance_id for f in probe.fragments
        ],
        "is_fragment_supported": probe.is_fragment_supported,
        "is_event_supported": probe.is_event_supported,
        "boundary_event_pairs": sorted(
            [list(p) for p in probe.boundary_event_pairs]
        ),
        "bridge_context": sorted(probe.bridge_context),
    }


def write_probes_jsonl(probes: Iterable[Probe], path: str | Path) -> int:
    """Write one Probe per line."""
    n = 0
    with Path(path).open("w") as fh:
        for p in probes:
            fh.write(json.dumps(probe_to_dict(p)))
            fh.write("\n")
            n += 1
    return n


# ---------- Responses ----------


def response_to_dict(response: RecallResponse) -> dict[str, Any]:
    """
    Serialize a RecallResponse to a JSON-friendly dict.

    Includes:
    - probe_id, probe_type, abstained, recall, fragment_provenance_ids
    - value_view (CRITICAL: without this, re-loaded responses default to
      TENANT_VISIBLE and silently score wrong if originally tenant-hidden)
    - rung, model, seed, tenant_visible (experiment metadata)
    - raw_text, parse_error, abstain_reason (debugging fields)

    Optional metadata fields are omitted from the output when None to keep
    the JSONL compact, EXCEPT value_view which is always serialized — it
    affects scoring semantics, not just metadata.
    """
    out: dict[str, Any] = {
        "probe_id": response.probe_id,
        "probe_type": response.probe_type.value,
        "abstained": response.abstained,
        "recall": recall_to_dict(response.recall) if response.recall else None,
        "fragment_provenance_ids": [
            f.public_provenance_id for f in response.fragments
        ],
        "value_view": response.value_view.value,
    }
    # Optional metadata: only emit when set.
    if response.rung is not None:
        out["rung"] = response.rung
    if response.model is not None:
        out["model"] = response.model
    if response.seed is not None:
        out["seed"] = response.seed
    if response.tenant_visible is not None:
        out["tenant_visible"] = response.tenant_visible
    if response.raw_text is not None:
        out["raw_text"] = response.raw_text
    if response.parse_error is not None:
        out["parse_error"] = response.parse_error
    if response.abstain_reason is not None:
        out["abstain_reason"] = response.abstain_reason
    return out


def write_responses_jsonl(
    responses: Iterable[RecallResponse], path: str | Path
) -> int:
    """Write one RecallResponse per line."""
    n = 0
    with Path(path).open("w") as fh:
        for r in responses:
            fh.write(json.dumps(response_to_dict(r)))
            fh.write("\n")
            n += 1
    return n


# ---------- Metrics ----------


def metric_summary_to_dict(summary: MetricSummary) -> dict[str, Any]:
    return {
        "boundary_invalid_recall_rate": summary.boundary_invalid_recall_rate,
        "fragment_valid_false_recall_rate": summary.fragment_valid_false_recall_rate,
        "supported_recall_accuracy": summary.supported_recall_accuracy,
        "absent_refusal_rate": summary.absent_refusal_rate,
        "partial_refusal_rate": summary.partial_refusal_rate,
        "audit_invisibility_per_verifier": dict(
            summary.audit_invisibility_per_verifier
        ),
        "n_responses": summary.n_responses,
        "n_violations": summary.n_violations,
    }


def write_metrics_json(summary: MetricSummary, path: str | Path) -> None:
    Path(path).write_text(json.dumps(metric_summary_to_dict(summary), indent=2))


# ---------- Rung statistics ----------


def rung_summary_to_dict(s: RungSummary) -> dict[str, Any]:
    return {
        "rung": s.rung.value,
        "n_probes": s.n_probes,
        "avg_fragments_retrieved": s.avg_fragments_retrieved,
        "avg_fragments_excluded": s.avg_fragments_excluded,
        "retention_by_probe_type": dict(s.retention_by_probe_type),
    }


def write_rung_stats_jsonl(
    summaries: Iterable[RungSummary], path: str | Path
) -> int:
    n = 0
    with Path(path).open("w") as fh:
        for s in summaries:
            fh.write(json.dumps(rung_summary_to_dict(s)))
            fh.write("\n")
            n += 1
    return n


# ---------- Read helpers (for replay/re-scoring) ----------


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Read a JSONL file into a list of dicts.

    Generic helper. Type-specific reconstruction (e.g., dict -> Probe) is
    intentionally NOT provided here, because reconstructing a Probe
    requires looking up Fragment objects by ID, which in turn requires
    the corpus. For most replay workflows, just regenerate the corpus
    from its seed.
    """
    out: list[dict[str, Any]] = []
    with Path(path).open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---------- Convenience: write the full experiment ----------


def write_experiment_artifacts(
    corpus: Corpus,
    responses: Iterable[RecallResponse],
    metric_summary: MetricSummary,
    rung_summaries: Iterable[RungSummary],
    out_dir: str | Path,
) -> dict[str, str]:
    """
    Write every artifact for one experiment run.

    Output layout under out_dir/:
        corpus.json
        records.jsonl
        fragments.tenant_visible.jsonl
        fragments.tenant_hidden.jsonl
        fragments.store.jsonl
        probes.jsonl
        responses.jsonl
        metrics.json
        rung_stats.jsonl

    Returns a dict of {artifact_name -> absolute_path}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    p = out_dir / "corpus.json"
    write_corpus_metadata(corpus, p)
    paths["corpus_metadata"] = str(p)

    p = out_dir / "records.jsonl"
    write_records_jsonl(corpus.records, p)
    paths["records"] = str(p)

    p = out_dir / "fragments.tenant_visible.jsonl"
    write_fragments_tenant_visible_jsonl(corpus, p)
    paths["fragments_tenant_visible"] = str(p)

    p = out_dir / "fragments.tenant_hidden.jsonl"
    write_fragments_tenant_hidden_jsonl(corpus, p)
    paths["fragments_tenant_hidden"] = str(p)

    p = out_dir / "fragments.store.jsonl"
    write_fragments_store_jsonl(corpus, p)
    paths["fragments_store"] = str(p)

    p = out_dir / "probes.jsonl"
    write_probes_jsonl(corpus.probes, p)
    paths["probes"] = str(p)

    p = out_dir / "responses.jsonl"
    write_responses_jsonl(responses, p)
    paths["responses"] = str(p)

    p = out_dir / "metrics.json"
    write_metrics_json(metric_summary, p)
    paths["metrics"] = str(p)

    p = out_dir / "rung_stats.jsonl"
    write_rung_stats_jsonl(rung_summaries, p)
    paths["rung_stats"] = str(p)

    return paths
