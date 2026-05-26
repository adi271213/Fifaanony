"""
fig_robustness — redesigned for USENIX/S&P/NeurIPS register.
Gray narrow bars, Wilson 95% CIs, Times 7pt, with x/y axes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# Use the canonical HARDENED scorer from rescore_final_ci.py
sys.path.insert(0, str(Path(__file__).parent))
from rescore_final_ci import boundary_invalid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams


BAR        = "#8A8A8E"
BAR_EMPH   = "#3A3A3C"
ERR        = "#1C1C1E"
GRID       = "#D9D9DE"
TEXT       = "#1C1C1E"
TEXT_SOFT  = "#5A5A60"
CHANCE     = "#C0392B"

FS_BODY  = 7.0
FS_TICK  = 6.5
FS_PANEL = 8.0
FS_VALUE = 6.5
FS_N     = 5.8

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
    "axes.linewidth": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.color": "#1C1C1E",
    "ytick.color": "#1C1C1E",
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.bottom": True,
    "axes.spines.left":   True,
    "axes.edgecolor":     "#1C1C1E",
})


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half   = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def load_counts(data_dir: Path):
    probes_by_cond, frags_by_cond = {}, {}
    for cd in sorted(data_dir.iterdir()):
        if not cd.is_dir():
            continue
        probes_by_cond[cd.name] = {}
        frags_by_cond[cd.name] = {}
        with (cd / "probes.jsonl").open() as f:
            for line in f:
                p = json.loads(line)
                probes_by_cond[cd.name][p["probe_id"]] = p
        with (cd / "fragments.store.jsonl").open() as f:
            for line in f:
                d = json.loads(line)
                frags_by_cond[cd.name][d["public_provenance_id"]] = d

    responses = []
    for cd in sorted(data_dir.iterdir()):
        if not cd.is_dir():
            continue
        with (cd / "responses.jsonl").open() as f:
            for line in f:
                r = json.loads(line)
                r["_condition"] = cd.name
                responses.append(r)

    cb_unsafe = [
        r for r in responses
        if r["probe_type"] == "cross_boundary"
        and r["rung"] in ("shared_vector_index", "summary_merge")
    ]

    def cat(b):
        p = b.split(":", 1)[0] if ":" in b else b
        return {
            "ip":     "Shared IP",
            "hash":   "Malware hash",
            "cve":    "CVE",
            "vendor": "Vendor",
            "actor":  "Threat actor",
        }.get(p, p)

    by_cat  = defaultdict(lambda: {"k": 0, "n": 0})
    by_span = defaultdict(lambda: {"k": 0, "n": 0})

    for r in cb_unsafe:
        pp = probes_by_cond[r["_condition"]].get(r["probe_id"])
        if not pp:
            continue
        # Use HARDENED boundary_invalid scorer from rescore_final_ci.py
        # to match the canonical scoring used in paper Tables and CSVs.
        store = frags_by_cond[r["_condition"]]
        hit = boundary_invalid(r, store)

        if pp.get("bridge_context"):
            c = cat(pp["bridge_context"][0])
            by_cat[c]["n"] += 1
            if hit:
                by_cat[c]["k"] += 1

        fmap = frags_by_cond[r["_condition"]]
        bds = set()
        for fp in pp.get("fragment_provenance_ids", []):
            if fp in fmap:
                bds.add(fmap[fp]["boundary_id"])
        span = len(bds)
        by_span[span]["n"] += 1
        if hit:
            by_span[span]["k"] += 1

    return by_cat, by_span


def opacity_for_n(n, n_max, lo=0.45, hi=1.0):
    if n_max <= 0:
        return hi
    return lo + (hi - lo) * math.sqrt(n / n_max)


def fmt_pct(p):
    v = 100 * p
    if abs(v - round(v)) < 0.05:
        return f"{int(round(v))}"
    return f"{v:.1f}"


def panel_horizontal(ax, labels, ks, ns, chance=None):
    n_max = max(ns) if ns else 1
    ys = list(range(len(labels)))[::-1]

    for x in (0.25, 0.5, 0.75, 1.0):
        ax.axvline(x, color=GRID, linewidth=0.5, zorder=1)
    if chance is not None:
        ax.axvline(chance, color=CHANCE, linewidth=0.7, linestyle=(0, (3, 2)),
                   zorder=1.5, alpha=0.8)

    for y, lab, k, n in zip(ys, labels, ks, ns):
        p, lo, hi = wilson_ci(k, n)
        emph = (k == n and n > 0)
        col = BAR_EMPH if emph else BAR
        alpha = opacity_for_n(n, n_max)
        ax.barh(y, p, height=0.35, color=col, alpha=alpha, edgecolor="none", zorder=2)
        ax.plot([lo, hi], [y, y], color=ERR, linewidth=0.8, zorder=3, solid_capstyle="butt")
        ax.plot([lo, lo], [y - 0.12, y + 0.12], color=ERR, linewidth=0.8, zorder=3)
        ax.plot([hi, hi], [y - 0.12, y + 0.12], color=ERR, linewidth=0.8, zorder=3)
        ax.text(1.04, y, f"{fmt_pct(p)}%", va="center", ha="left",
                fontsize=FS_VALUE, color=TEXT,
                fontweight="bold" if emph else "normal")
        ax.text(1.22, y, f"n={n}", va="center", ha="left",
                fontsize=FS_N, color=TEXT_SOFT)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=FS_TICK, color=TEXT)
    ax.tick_params(axis="y", length=2.5, pad=3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=FS_TICK, color=TEXT_SOFT)
    ax.tick_params(axis="x", length=2.5, pad=3)
    ax.set_xlim(0, 1.32)
    ax.set_ylim(-0.6, len(labels) - 0.4)


def panel_vertical(ax, labels, ks, ns, chance=None, xlab=""):
    n_max = max(ns) if ns else 1
    xs = list(range(len(labels)))

    for y in (0.25, 0.5, 0.75, 1.0):
        ax.axhline(y, color=GRID, linewidth=0.5, zorder=1)
    if chance is not None:
        ax.axhline(chance, color=CHANCE, linewidth=0.7, linestyle=(0, (3, 2)),
                   zorder=1.5, alpha=0.8)

    for x, lab, k, n in zip(xs, labels, ks, ns):
        p, lo, hi = wilson_ci(k, n)
        emph = (k == n and n > 0)
        col = BAR_EMPH if emph else BAR
        alpha = opacity_for_n(n, n_max)
        ax.bar(x, p, width=0.30, color=col, alpha=alpha, edgecolor="none", zorder=2)
        ax.plot([x, x], [lo, hi], color=ERR, linewidth=0.8, zorder=3, solid_capstyle="butt")
        ax.plot([x - 0.10, x + 0.10], [lo, lo], color=ERR, linewidth=0.8, zorder=3)
        ax.plot([x - 0.10, x + 0.10], [hi, hi], color=ERR, linewidth=0.8, zorder=3)
        ax.text(x, min(hi + 0.04, 1.08), f"{fmt_pct(p)}%",
                ha="center", va="bottom",
                fontsize=FS_VALUE, color=TEXT,
                fontweight="bold" if emph else "normal")
        ax.text(x, -0.07, f"n={n}", ha="center", va="top",
                fontsize=FS_N, color=TEXT_SOFT)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=FS_TICK, color=TEXT)
    ax.tick_params(axis="x", length=2.5, pad=13)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"], fontsize=FS_TICK, color=TEXT_SOFT)
    ax.tick_params(axis="y", length=2.5, pad=3)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 1.18)
    if xlab:
        ax.set_xlabel(xlab, fontsize=FS_BODY, color=TEXT, labelpad=14)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out",  type=Path, default=Path("/tmp/fig_previews/fig_robustness.pdf"))
    ap.add_argument("--chance", type=float, default=None)
    args = ap.parse_args()

    by_cat, by_span = load_counts(args.data)

    cat_order = ["Shared IP", "Malware hash", "CVE", "Vendor", "Threat actor"]
    cat_ks = [by_cat[c]["k"] for c in cat_order]
    cat_ns = [by_cat[c]["n"] for c in cat_order]

    span_keys = sorted(by_span.keys())
    span_lbls = [str(s) for s in span_keys]
    span_ks   = [by_span[s]["k"] for s in span_keys]
    span_ns   = [by_span[s]["n"] for s in span_keys]

    fig = plt.figure(figsize=(3.3, 4.6), dpi=300)
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        left=0.22, right=0.95, top=0.91, bottom=0.10,
        hspace=0.95,
        height_ratios=[len(cat_order), max(2, len(span_keys))],
    )
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    panel_horizontal(ax_a, cat_order, cat_ks, cat_ns, chance=args.chance)
    panel_vertical(  ax_b, span_lbls, span_ks, span_ns, chance=args.chance,
                     xlab="Distinct boundaries spanned")

    ax_a.set_xlabel("Boundary-invalid recall rate", fontsize=FS_BODY, color=TEXT, labelpad=4)
    ax_b.set_ylabel("Boundary-invalid recall rate", fontsize=FS_BODY, color=TEXT, labelpad=4)

    fig.text(0.04, 0.955, "(a)", fontsize=FS_PANEL, fontweight="bold", color=TEXT)
    fig.text(0.10, 0.955, "Failure rate by bridge category", fontsize=FS_BODY, color=TEXT)
    fig.text(0.04, 0.475, "(b)", fontsize=FS_PANEL, fontweight="bold", color=TEXT)
    fig.text(0.10, 0.475, "Failure rate by boundary span",   fontsize=FS_BODY, color=TEXT)

    if args.chance is not None:
        fig.text(0.95, 0.955,
                 f"– – – chance = {int(round(100*args.chance))}%",
                 fontsize=FS_N, color=CHANCE, ha="right")

    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    png = args.out.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", pad_inches=0.05, facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Wrote {args.out} and {png}")


if __name__ == "__main__":
    main()
