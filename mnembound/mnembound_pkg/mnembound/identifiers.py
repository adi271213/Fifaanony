"""
mnembound.identifiers — Per-tenant identifier pool construction.

Enforces the central invariant:

    Entity, boundary, and event identifiers are DISJOINT across tenants by
    construction.

- Boundary granularity is now configurable. Default is TENANT (one
  confidentiality boundary per tenant), matching the paper's MSSP framing.
  An earlier version implicitly used EVENT granularity (one boundary per record), which
  conflated boundary-level violations with within-tenant event-level
  inconsistencies and blurred the paper's thesis.
- Added view-sanitization helpers (`build_entity_sanitizer`, `sanitize_text`)
  so the V_frag view can be made truly tenant-hidden when needed for the
  §7 baseline-fairness comparison.
"""

from __future__ import annotations

import hashlib
import random
import string
from dataclasses import dataclass
from enum import Enum
from typing import Any


_TENANT_PREFIX_ALPHABET = string.ascii_uppercase
_N_LETTERS = len(_TENANT_PREFIX_ALPHABET)


class BoundaryGranularity(str, Enum):
    """
    How confidentiality boundaries are scoped within a tenant.

    TENANT     (default): one boundary per tenant. All of clientA's records
                          share boundary_id 'b-A'. Matches the MSSP framing
                          where the confidentiality boundary is the tenant.
    ENGAGEMENT:           a fixed number of boundaries per tenant. Records
                          are partitioned across boundaries within a tenant.
                          Useful for modelling per-engagement scopes in an
                          MSSP that runs many engagements per client.
    EVENT:                one boundary per record. Legacy default, kept for
                          backwards-compat and stress-testing. NOT the right
                          semantics for the headline paper claim.
    """

    TENANT = "tenant"
    ENGAGEMENT = "engagement"
    EVENT = "event"


def tenant_prefix(tenant_index: int) -> str:
    """
    Map a non-negative tenant index to a unique alphabetic prefix.

    0..25      -> A..Z         (1 letter)
    26..701    -> AA..ZZ       (2 letters)
    702..17577 -> AAA..ZZZ     (3 letters)
    etc.

    Bijective base-26 expansion.
    """
    if tenant_index < 0:
        raise ValueError(f"tenant_index must be non-negative, got {tenant_index}")

    n = _N_LETTERS
    tier = 1
    tier_start = 0
    tier_count = n
    while tenant_index >= tier_start + tier_count:
        tier_start += tier_count
        tier += 1
        tier_count *= n

    offset = tenant_index - tier_start
    letters: list[str] = []
    for _ in range(tier):
        offset, digit = divmod(offset, n)
        letters.append(_TENANT_PREFIX_ALPHABET[digit])
    return "".join(reversed(letters))


def tenant_id(tenant_index: int) -> str:
    """The tenant string identifier (e.g., 'clientA', 'clientB')."""
    return f"client{tenant_prefix(tenant_index)}"


@dataclass(frozen=True)
class TenantEntityPool:
    """Per-tenant entity pool. Disjoint from every other tenant's pool."""

    tenant_index: int
    tenant_id: str
    hosts: frozenset[str]
    users: frozenset[str]
    files: frozenset[str]
    processes: frozenset[str]
    malware_families: frozenset[str]

    def all_entities(self) -> frozenset[str]:
        return (
            self.hosts
            | self.users
            | self.files
            | self.processes
            | self.malware_families
        )


def build_tenant_entity_pools(
    n_tenants: int,
    seed: int,
    pool_size_hosts: int = 200,
    pool_size_users: int = 100,
    pool_size_files: int = 500,
    pool_size_processes: int = 200,
    pool_size_malware: int = 50,
) -> dict[str, TenantEntityPool]:
    """Build n_tenants disjoint entity pools."""
    if n_tenants < 1:
        raise ValueError(f"n_tenants must be at least 1, got {n_tenants}")

    rng = random.Random(seed)
    pools: dict[str, TenantEntityPool] = {}

    for ti in range(n_tenants):
        tprefix = tenant_prefix(ti)
        tid = tenant_id(ti)

        hosts = frozenset(
            f"host:{name}-{tprefix}.{tid}.local"
            for name in _sample_words(rng, pool_size_hosts, "host")
        )
        users = frozenset(
            f"user:{name}-{tprefix}@{tid}.local"
            for name in _sample_words(rng, pool_size_users, "user")
        )
        files = frozenset(
            f"file:/var/data/{tprefix}/{name}-{i:04d}.bin"
            for i, name in enumerate(_sample_words(rng, pool_size_files, "file"))
        )
        processes = frozenset(
            f"proc:{name}-{tprefix}-{i:04d}.exe"
            for i, name in enumerate(_sample_words(rng, pool_size_processes, "proc"))
        )
        malware = frozenset(
            f"malware:{name}_{tprefix}"
            for name in _sample_words(rng, pool_size_malware, "malware")
        )

        pools[tid] = TenantEntityPool(
            tenant_index=ti,
            tenant_id=tid,
            hosts=hosts,
            users=users,
            files=files,
            processes=processes,
            malware_families=malware,
        )

    return pools


def _sample_words(rng: random.Random, n: int, kind: str) -> list[str]:
    """Produce n unique deterministic word stems."""
    vocabularies = {
        "host": [
            "web", "db", "auth", "api", "cache", "queue", "worker", "edge",
            "proxy", "build", "stage", "ml", "etl", "log", "monitor", "ldap",
            "smtp", "dns", "vpn", "bastion",
        ],
        "user": [
            "alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi",
            "ivan", "judy", "ken", "leah", "mallory", "ned", "olivia", "peggy",
            "quinn", "rita", "sam", "trent",
        ],
        "file": [
            "config", "creds", "report", "export", "backup", "tmp", "cache",
            "spool", "audit", "log", "session", "token", "key", "dump",
            "snapshot", "delta", "patch", "package", "manifest", "ledger",
        ],
        "proc": [
            "service", "agent", "daemon", "worker", "scanner", "updater",
            "monitor", "indexer", "collector", "broker", "scheduler",
            "watchdog", "syncer", "compactor", "validator", "rotator",
            "shipper", "ingest", "compiler", "linker",
        ],
        "malware": [
            "Cobra", "Ferret", "Glider", "Hawk", "Iguana", "Jackal", "Kestrel",
            "Lemur", "Marten", "Newt", "Orca", "Puma", "Quagga", "Raven",
            "Stoat", "Tarsier", "Urial", "Viper", "Wolverine", "Xerus",
        ],
    }
    base = vocabularies[kind]
    shuffled = list(base)
    rng.shuffle(shuffled)
    out: list[str] = []
    counter = 0
    while len(out) < n:
        for word in shuffled:
            out.append(f"{word}{counter:03d}" if counter > 0 else word)
            if len(out) >= n:
                break
        counter += 1
    return out[:n]


# ----- Boundary and event identifier builders -----


def boundary_id_for(
    tenant_index: int,
    record_local: int,
    granularity: BoundaryGranularity,
    n_engagements_per_tenant: int = 5,
) -> str:
    """
    Compute the boundary_id for a record given the configured granularity.

    TENANT      -> one boundary per tenant.            'b-A'
    ENGAGEMENT  -> n_engagements_per_tenant per tenant. 'b-A-eng002'
    EVENT       -> one boundary per record.            'b-A-evt000042'
    """
    tprefix = tenant_prefix(tenant_index)
    if granularity is BoundaryGranularity.TENANT:
        return f"b-{tprefix}"
    if granularity is BoundaryGranularity.ENGAGEMENT:
        if n_engagements_per_tenant < 1:
            raise ValueError(
                f"n_engagements_per_tenant must be at least 1, got "
                f"{n_engagements_per_tenant}"
            )
        engagement = record_local % n_engagements_per_tenant
        return f"b-{tprefix}-eng{engagement:03d}"
    if granularity is BoundaryGranularity.EVENT:
        return f"b-{tprefix}-evt{record_local:06d}"
    raise ValueError(f"unknown boundary granularity: {granularity}")


def make_event_id(tenant_index: int, event_local: int) -> str:
    """Construct a tenant-namespaced event identifier (e.g., 'e-A-000042')."""
    return f"e-{tenant_prefix(tenant_index)}-{event_local:06d}"


def make_record_id(tenant_index: int, record_local: int) -> str:
    """Construct a tenant-namespaced record identifier (e.g., 'r-A-000000')."""
    return f"r-{tenant_prefix(tenant_index)}-{record_local:06d}"


# ----- View sanitization for tenant-hidden mode -----
#
# The V_frag baselines must, in some experimental conditions, see entity
# strings that do NOT reveal tenant identity. Per §4.4 baseline fairness,
# §7 will run BOTH tenant-hidden and tenant-visible source-set entailment
# to isolate "did NLI fail because tenant labels were absent" from "did NLI
# fail because the entailment problem is genuinely hard."
#
# The sanitizer maps a tenant-revealing entity like
#     'host:web001-A.clientA.local'
# to a stable, tenant-hidden token like
#     'host:h_a8f3c12b'
#
# Properties:
# 1. Deterministic given (entity, salt). Same input -> same output.
# 2. Stable within a corpus (the corpus's provenance salt is reused), so
#    audit-trail replay against sanitized text behaves correctly.
# 3. Reversible only by a verifier that holds the salt and the inverse map
#    (i.e., a store-level verifier). V_frag baselines do not get the inverse.
# 4. Preserves the entity type prefix (host, user, file, proc, malware) so
#    NLI baselines can still infer that a fragment refers to a host vs. a
#    user. This is the right level of information leakage: the §7 comparison
#    isolates tenant identity, not entity type.


_ENTITY_TYPE_PREFIXES: tuple[str, ...] = (
    "host:",
    "user:",
    "file:",
    "proc:",
    "malware:",
)


_TYPE_LETTER: dict[str, str] = {
    "host:": "h",
    "user:": "u",
    "file:": "f",
    "proc:": "p",
    "malware:": "m",
}


def _entity_type_prefix(entity: str) -> tuple[str, str]:
    """
    Split an entity string into (type_prefix, rest).

        'host:web001-A.clientA.local' -> ('host:', 'web001-A.clientA.local')
        'plain_string'                 -> ('', 'plain_string')
    """
    for prefix in _ENTITY_TYPE_PREFIXES:
        if entity.startswith(prefix):
            return prefix, entity[len(prefix):]
    return "", entity


def _opaque_token_short(letter: str, body: str, salt: str, extra_bytes: int = 0) -> str:
    """
    Build a short opaque token body for an entity under a given salt.

    Returns the token body only (e.g., 'h_a8f3c12b'). Callers prepend the
    entity-type prefix.

    8 hex chars = 32 bits. Combined with the entity type prefix, collisions
    are unlikely at typical scales (~500 entities/tenant × 10 tenants
    = 5000). The sanitizer builder runs an explicit collision check and
    widens via `extra_bytes` if a collision is detected.
    """
    raw = f"{salt}|{letter}|{body}|{extra_bytes}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[: 8 + extra_bytes]
    return f"{letter}_{digest}"


def build_entity_sanitizer(
    pools: dict[str, TenantEntityPool],
    salt: str,
) -> dict[str, str]:
    """
    Build a one-to-one map from every per-tenant entity string to a
    tenant-hidden opaque token.

    The token preserves entity-type information (`host:` -> `host:h_...`,
    `user:` -> `user:u_...`, etc.) so an NLI baseline can still tell that
    a fragment refers to a host vs. a user. It does NOT preserve tenant
    identity.

    Collisions are checked. If two distinct entities collide, the token
    width for the colliding entity is widened by 8 hex chars and retried.

    Returns: dict { original_entity_string -> sanitized_token }.
    """
    sanitizer: dict[str, str] = {}
    token_to_source: dict[str, str] = {}

    # Iterate pools in deterministic order so the sanitizer is reproducible.
    for tid in sorted(pools.keys()):
        pool = pools[tid]
        for entity in sorted(pool.all_entities()):
            prefix, body = _entity_type_prefix(entity)
            letter = _TYPE_LETTER.get(prefix, "x")

            extra_bytes = 0
            token_body = _opaque_token_short(letter, body, salt, extra_bytes)
            token = f"{prefix}{token_body}"

            while token in token_to_source and token_to_source[token] != entity:
                extra_bytes += 8
                token_body = _opaque_token_short(letter, body, salt, extra_bytes)
                token = f"{prefix}{token_body}"
                if extra_bytes > 64:
                    raise RuntimeError(
                        f"sanitizer collision could not be resolved for entity "
                        f"{entity!r} after expanding to 72 hex chars; this "
                        "indicates a deeper sanitizer bug, not chance collision"
                    )

            sanitizer[entity] = token
            token_to_source[token] = entity

    return sanitizer


def sanitize_text(text: str, sanitizer: dict[str, str]) -> str:
    """
    Replace every per-tenant entity occurrence in `text` with its sanitized
    token. Other content (verbs, timestamps, log sources) is unchanged.

    Convenience wrapper that builds the regex on each call. For corpus-scale
    use, the generator builds a single regex via `compile_sanitizer_regex`
    and reuses it across all fragments.
    """
    pattern, _ = compile_sanitizer_regex(sanitizer)
    return sanitize_text_with_compiled_regex(text, sanitizer, pattern)


def compile_sanitizer_regex(sanitizer: dict[str, str]) -> tuple[Any, list[str]]:
    """
    Compile a single regex alternation matching every sanitizer key.

    Keys are sorted by descending length so longer entity strings are
    preferred over substrings. The compiled regex is used by
    sanitize_text_with_compiled_regex for performance.

    Returns: (compiled_pattern, keys_by_length).
    """
    import re as _re
    keys_by_length = sorted(sanitizer.keys(), key=len, reverse=True)
    if not keys_by_length:
        # Empty sanitizer: a regex that never matches.
        pattern = _re.compile(r"(?!x)x")
    else:
        pattern = _re.compile(
            "|".join(_re.escape(k) for k in keys_by_length)
        )
    return pattern, keys_by_length


def sanitize_text_with_compiled_regex(
    text: str,
    sanitizer: dict[str, str],
    pattern: Any,
) -> str:
    """
    Performance-optimized form of sanitize_text using a precompiled regex.

    The generator builds the regex once and uses this across every fragment.
    """
    return pattern.sub(lambda m: sanitizer[m.group(0)], text)


def sanitize_text_with_sorted_keys(
    text: str,
    sanitizer: dict[str, str],
    keys_by_length: list[str],
) -> str:
    """
    Iterative form (used by some callers).

    Prefer sanitize_text_with_compiled_regex for new code.
    """
    out = text
    for key in keys_by_length:
        if key in out:
            out = out.replace(key, sanitizer[key])
    return out
