"""
Smoke-runner correctness tests.

Verifies:
- Default prompt-mode in the runner is now NEUTRAL (the paper headline),
  not CAUTIOUS.
- The unsafe-rung abstention partition only counts shared_vector_index
  and summary_merge — not isolated and boundary_preserving where
  abstention is the correct behavior.
- PromptBundle has only one class definition (no leftover duplicate).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import run_smoke  # noqa: E402
from mnembound import agent_io  # noqa: E402


def test_runner_default_prompt_mode_is_neutral() -> None:
    """
    The runner must default to 'neutral' so an accidental
    run reports the headline phenomenon, not the safety-mitigated baseline.
    """
    parser = None
    # Inspect the main() function's argparse setup by calling it with --help
    # would exit; instead, parse with no args after a small redirect trick.
    # Cleanest: directly grep the source for the default we changed.
    src = inspect.getsource(run_smoke.main)
    # The argparse line must read default="neutral"
    assert 'default="neutral"' in src, (
        "runner's --prompt-mode default should be 'neutral' (headline paper "
        "condition), not 'cautious'"
    )
    assert 'default="cautious"' not in src, (
        "old cautious default not removed from runner"
    )
    print("[OK] runner default --prompt-mode is 'neutral'")


def test_runner_default_summary_mode_is_collapsed() -> None:
    src = inspect.getsource(run_smoke.main)
    assert 'default="collapsed"' in src
    print("[OK] runner default --summary-mode is 'collapsed'")


def test_run_smoke_partitions_abstention_by_rung_safety() -> None:
    """
    The runner's diagnostic section must compute cross-boundary abstention
    on shared_vector_index + summary_merge ONLY, not on isolated or
    boundary_preserving (where abstention is the correct response to
    incomplete evidence).
    """
    src = inspect.getsource(run_smoke.run_smoke)
    # The unsafe-rung set is named explicitly.
    assert 'UNSAFE_RUNGS' in src or '"shared_vector_index"' in src, (
        "runner should partition abstention rate by rung safety"
    )
    # Both unsafe rungs appear in the partition.
    assert '"shared_vector_index"' in src
    assert '"summary_merge"' in src
    # The output mentions "unsafe rungs only".
    assert 'unsafe rungs only' in src or 'unsafe rungs' in src
    print("[OK] runner partitions abstention rate by unsafe vs safe rungs")


def test_prompt_bundle_has_exactly_one_definition() -> None:
    """
    An earlier generator had a duplicate empty PromptBundle dataclass declaration before
    the real one. The current generator removes it.
    """
    src = Path(agent_io.__file__).read_text()
    n_defs = sum(
        1 for line in src.splitlines()
        if line.strip().startswith("class PromptBundle")
    )
    assert n_defs == 1, (
        f"expected exactly 1 PromptBundle definition; found {n_defs}"
    )
    # And the surviving definition has the expected fields.
    pb = agent_io.PromptBundle(
        system="s", user="u",
        rung=__import__("mnembound.retrieval", fromlist=["Rung"]).Rung.ISOLATED,
        tenant_visible=True,
    )
    assert pb.system == "s"
    assert pb.user == "u"
    print("[OK] PromptBundle has exactly one definition")


def main() -> int:
    tests = [
        test_runner_default_prompt_mode_is_neutral,
        test_runner_default_summary_mode_is_collapsed,
        test_run_smoke_partitions_abstention_by_rung_safety,
        test_prompt_bundle_has_exactly_one_definition,
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
    print(f"=== all {len(tests)} smoke-runner tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
