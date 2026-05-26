"""
mnembound.templates — SOC incident templates.

Each template is a recipe for producing a Record from an entity pool. The
template defines:

- which entity types fill which recall components
- which log_source typically reports this incident type
- the per-component sentence text a fragment-level verifier would see

Templates own the rendering. Records store the rendered text. Fragments
read the text from the Record. There is exactly one place where text is
constructed, eliminating drift risk between templates.py and
generator.py.

We provide five templates spanning the most common SOC incident classes:

1. INTRUSION           — initial access via credential compromise
2. EXFILTRATION        — data egress to an external destination
3. RANSOMWARE          — file encryption with ransom note
4. ACCOUNT_COMPROMISE  — account takeover and privilege escalation
5. MALWARE_DETONATION  — malicious executable launched on host
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping

from mnembound.identifiers import TenantEntityPool
from mnembound.schema import ComponentRole


class TemplateName(str, Enum):
    INTRUSION = "intrusion_v1"
    EXFILTRATION = "exfiltration_v1"
    RANSOMWARE = "ransomware_v1"
    ACCOUNT_COMPROMISE = "account_compromise_v1"
    MALWARE_DETONATION = "malware_detonation_v1"


TEMPLATE_LOG_SOURCES: dict[TemplateName, tuple[str, ...]] = {
    TemplateName.INTRUSION: ("siem:Splunk", "auth:Okta", "edr:CrowdStrike"),
    TemplateName.EXFILTRATION: ("dlp:Forcepoint", "firewall:Palo", "siem:Splunk"),
    TemplateName.RANSOMWARE: ("edr:CrowdStrike", "edr:SentinelOne", "av:Defender"),
    TemplateName.ACCOUNT_COMPROMISE: ("auth:Okta", "auth:AzureAD", "siem:Splunk"),
    TemplateName.MALWARE_DETONATION: ("edr:CrowdStrike", "av:Defender", "siem:Splunk"),
}

DEFAULT_TEMPLATE_WEIGHTS: dict[TemplateName, float] = {
    TemplateName.INTRUSION: 0.20,
    TemplateName.EXFILTRATION: 0.15,
    TemplateName.RANSOMWARE: 0.10,
    TemplateName.ACCOUNT_COMPROMISE: 0.30,
    TemplateName.MALWARE_DETONATION: 0.25,
}


@dataclass(frozen=True)
class TemplateOutput:
    """The result of instantiating one template into the fields a Record needs."""

    subject: str
    action: str
    object_: str
    outcome: str
    time: str
    log_source: str
    template: TemplateName
    extra: Mapping[str, str]

    # Single source of truth for per-component sentence text.
    component_text: Mapping[ComponentRole, str]


TemplateRenderer = Callable[[random.Random, TenantEntityPool, datetime], TemplateOutput]


# ----- Template implementations -----

def _render_intrusion(
    rng: random.Random, pool: TenantEntityPool, timestamp: datetime
) -> TemplateOutput:
    user = rng.choice(sorted(pool.users))
    host = rng.choice(sorted(pool.hosts))
    outcome = rng.choice(["success", "blocked", "detected"])
    time_str = timestamp.isoformat()
    log_source = rng.choice(TEMPLATE_LOG_SOURCES[TemplateName.INTRUSION])

    return TemplateOutput(
        subject=user,
        action="authenticated",
        object_=host,
        outcome=outcome,
        time=time_str,
        log_source=log_source,
        template=TemplateName.INTRUSION,
        extra=MappingProxyType({}),
        component_text=MappingProxyType({
            ComponentRole.SUBJECT: f"Authentication attempt by {user}.",
            ComponentRole.ACTION: f"Authentication action observed at {time_str}.",
            ComponentRole.OBJECT: f"Target host {host} received the authentication request.",
            ComponentRole.OUTCOME: f"Authentication result: {outcome}.",
            ComponentRole.TIME: f"Event timestamped {time_str}.",
        }),
    )


def _render_exfiltration(
    rng: random.Random, pool: TenantEntityPool, timestamp: datetime
) -> TemplateOutput:
    user = rng.choice(sorted(pool.users))
    file_ = rng.choice(sorted(pool.files))
    outcome = rng.choice(["blocked", "completed", "detected"])
    time_str = timestamp.isoformat()
    log_source = rng.choice(TEMPLATE_LOG_SOURCES[TemplateName.EXFILTRATION])

    return TemplateOutput(
        subject=user,
        action="exfiltrated",
        object_=file_,
        outcome=outcome,
        time=time_str,
        log_source=log_source,
        template=TemplateName.EXFILTRATION,
        extra=MappingProxyType({}),
        component_text=MappingProxyType({
            ComponentRole.SUBJECT: f"Egress event initiated by {user}.",
            ComponentRole.ACTION: f"Data exfiltration action recorded at {time_str}.",
            ComponentRole.OBJECT: f"File {file_} flagged in egress traffic.",
            ComponentRole.OUTCOME: f"Egress outcome: {outcome}.",
            ComponentRole.TIME: f"Egress event timestamped {time_str}.",
        }),
    )


def _render_ransomware(
    rng: random.Random, pool: TenantEntityPool, timestamp: datetime
) -> TemplateOutput:
    proc = rng.choice(sorted(pool.processes))
    host = rng.choice(sorted(pool.hosts))
    outcome = rng.choice(["contained", "encrypted", "partial"])
    time_str = timestamp.isoformat()
    log_source = rng.choice(TEMPLATE_LOG_SOURCES[TemplateName.RANSOMWARE])

    return TemplateOutput(
        subject=proc,
        action="encrypted_files_on",
        object_=host,
        outcome=outcome,
        time=time_str,
        log_source=log_source,
        template=TemplateName.RANSOMWARE,
        extra=MappingProxyType({}),
        component_text=MappingProxyType({
            ComponentRole.SUBJECT: f"Suspicious process {proc} observed.",
            ComponentRole.ACTION: f"File-encryption activity detected at {time_str}.",
            ComponentRole.OBJECT: f"Affected host: {host}.",
            ComponentRole.OUTCOME: f"Encryption-attempt outcome: {outcome}.",
            ComponentRole.TIME: f"Detection timestamp {time_str}.",
        }),
    )


def _render_account_compromise(
    rng: random.Random, pool: TenantEntityPool, timestamp: datetime
) -> TemplateOutput:
    user = rng.choice(sorted(pool.users))
    host = rng.choice(sorted(pool.hosts))
    outcome = rng.choice(["mfa_bypassed", "blocked", "session_revoked"])
    time_str = timestamp.isoformat()
    log_source = rng.choice(TEMPLATE_LOG_SOURCES[TemplateName.ACCOUNT_COMPROMISE])

    return TemplateOutput(
        subject=user,
        action="elevated_privileges_on",
        object_=host,
        outcome=outcome,
        time=time_str,
        log_source=log_source,
        template=TemplateName.ACCOUNT_COMPROMISE,
        extra=MappingProxyType({}),
        component_text=MappingProxyType({
            ComponentRole.SUBJECT: f"Account {user} flagged for anomalous activity.",
            ComponentRole.ACTION: f"Privilege-elevation action observed at {time_str}.",
            ComponentRole.OBJECT: f"Target resource: {host}.",
            ComponentRole.OUTCOME: f"Compromise-handling outcome: {outcome}.",
            ComponentRole.TIME: f"Event timestamped {time_str}.",
        }),
    )


def _render_malware_detonation(
    rng: random.Random, pool: TenantEntityPool, timestamp: datetime
) -> TemplateOutput:
    malware = rng.choice(sorted(pool.malware_families))
    host = rng.choice(sorted(pool.hosts))
    outcome = rng.choice(["quarantined", "executed", "blocked"])
    time_str = timestamp.isoformat()
    log_source = rng.choice(TEMPLATE_LOG_SOURCES[TemplateName.MALWARE_DETONATION])

    return TemplateOutput(
        subject=malware,
        action="executed_on",
        object_=host,
        outcome=outcome,
        time=time_str,
        log_source=log_source,
        template=TemplateName.MALWARE_DETONATION,
        extra=MappingProxyType({}),
        component_text=MappingProxyType({
            ComponentRole.SUBJECT: f"Malware signature {malware} matched.",
            ComponentRole.ACTION: f"Executable launched at {time_str}.",
            ComponentRole.OBJECT: f"Endpoint affected: {host}.",
            ComponentRole.OUTCOME: f"EDR response outcome: {outcome}.",
            ComponentRole.TIME: f"Detonation timestamp {time_str}.",
        }),
    )


TEMPLATE_RENDERERS: dict[TemplateName, TemplateRenderer] = {
    TemplateName.INTRUSION: _render_intrusion,
    TemplateName.EXFILTRATION: _render_exfiltration,
    TemplateName.RANSOMWARE: _render_ransomware,
    TemplateName.ACCOUNT_COMPROMISE: _render_account_compromise,
    TemplateName.MALWARE_DETONATION: _render_malware_detonation,
}


def pick_template(rng: random.Random) -> TemplateName:
    """Sample a template name according to DEFAULT_TEMPLATE_WEIGHTS."""
    names = list(DEFAULT_TEMPLATE_WEIGHTS.keys())
    weights = [DEFAULT_TEMPLATE_WEIGHTS[n] for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def render_template(
    template: TemplateName,
    rng: random.Random,
    pool: TenantEntityPool,
    timestamp: datetime,
) -> TemplateOutput:
    """Dispatch to the appropriate renderer."""
    return TEMPLATE_RENDERERS[template](rng, pool, timestamp)


def sample_timestamp(
    rng: random.Random,
    base: datetime,
    window_days: int = 90,
) -> datetime:
    """Sample a timestamp uniformly within [base, base + window_days]."""
    seconds_in_window = window_days * 24 * 60 * 60
    offset = rng.randint(0, seconds_in_window - 1)
    return base + timedelta(seconds=offset)
