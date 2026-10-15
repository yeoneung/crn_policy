"""Secondary replay: standard PM-EB versus fixed betting, with two supports.

Run with Python 3.11+, numpy, scipy, matplotlib. No training or dynamics calls.
The protocol is in ../replay_protocol.json; original assets are immutable.
"""
from pathlib import Path
import hashlib
import json
import math
from datetime import datetime, timezone
import numpy as np
from scipy.special import logsumexp
from return_certificate import FRACTIONS, WEIGHTS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "code/results/v2_variance_replay"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log_paths(d, bound, alpha, method):
    d = np.asarray(d, dtype=float)
    if (d.ndim != 1 or not np.isfinite(d).all() or not np.isfinite(bound)
            or bound <= 0 or not 0 < alpha < 1 or np.any(np.abs(d) > bound)):
        raise ValueError("Invalid inputs or violated support; never clip observations")
    if method == "fixed_mixture":
        with np.errstate(divide="ignore"):
            parts = np.cumsum(np.log1p(FRACTIONS[:, None] * d[None, :] / bound), axis=1)
        return logsumexp(np.log(WEIGHTS)[:, None] + parts, axis=0)
    if method != "predictable_empirical_bernstein":
        raise ValueError(method)
    # X in [0,1], testing E X <= 1/2. Each lambda and predictor use only
    # earlier observations. The regularized variance uses CURRENT means,
    # whereas the exponential penalty uses the PREVIOUS mean (WSR eq.16).
    x = (d / bound + 1) / 2
    t = np.arange(1, len(x) + 1, dtype=float)
    mean = (0.5 + np.cumsum(x)) / (t + 1)
    previous_mean = np.r_[0.5, mean[:-1]]
    variance = (0.25 + np.cumsum((x - mean)**2)) / (t + 1)
    previous_variance = np.r_[0.25, variance[:-1]]
    lam = np.minimum(0.75, np.sqrt(2 * np.log(2 / alpha) /
                                      (previous_variance * t * np.log1p(t))))
    psi = (-np.log1p(-lam) - lam) / 4
    return np.cumsum(lam * (x - 0.5) - 4 * (x - previous_mean)**2 * psi)


def stop(d, bound, alpha, method, cap, block=16):
    d = np.asarray(d)[:cap]
    if len(d) != cap:
        raise ValueError("Insufficient retained observations")
    plus = log_paths(d, bound, alpha, method)
    minus = log_paths(-d, bound, alpha, method)
    checkpoints = np.unique(np.r_[np.arange(block, cap + 1, block), cap]).astype(int)
    n, outcome = cap, "inconclusive"
    for k in checkpoints:
        if plus[k-1] >= np.log(1 / alpha):
            n, outcome = int(k), "adopt"
            break
        if minus[k-1] >= np.log(1 / alpha):
            n, outcome = int(k), "reject"
            break
    return dict(outcome=outcome, adopted=outcome == "adopt", pairs=n,
                positive_log_wealth=float(plus[n-1]), negative_log_wealth=float(minus[n-1]))


def replay():
    protocol = ROOT / "code/replay_protocol.json"
    manifest = {str(protocol.relative_to(ROOT)): digest(protocol),
                str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
    rows = []
    for study, support in [("coupled_support_confirmatory", "normalized_support"),
                           ("nonnormal_confirmatory", "matrix_power"),
                           ("actuation_structure_confirmatory", "input_structure")]:
        for directory in sorted((ROOT / "code/results" / study).glob("dim*_seed*")):
            paths = [directory / name for name in
                     ["pre_evaluation.json", "certificates.json", "certification.npz", "evaluation.npz"]]
            for p in paths:
                manifest[str(p.relative_to(ROOT))] = digest(p)
            certificates = json.loads(paths[1].read_text())
            pre = json.loads(paths[0].read_text())
            idx = pre["names"].index("Ridge_d0.05_b81920") if study == "coupled_support_confirmatory" else 1
            with np.load(paths[2]) as bank:
                d = bank["returns"][idx] - bank["returns"][0]
            with np.load(paths[3]) as bank:
                gain = float(np.mean(bank["returns"][idx] - bank["returns"][0]))
            for kind in (support, "ordinary_range"):
                for scope in ("primary", "extended"):
                    cap = 4096 if scope == "extended" else (512 if kind != "ordinary_range" or study == "coupled_support_confirmatory" else 716)
                    selected = [c for c in certificates if c["bound_kind"] == kind and c["cap"] == cap
                                and (study != "coupled_support_confirmatory" or c["name"] == "Ridge_d0.05_b81920")]
                    assert len(selected) == 1, (study, directory.name, kind, scope, len(selected))
                    old = selected[0]
                    for method in ("fixed_mixture", "predictable_empirical_bernstein"):
                        result = stop(d, old["bound"], old["alpha"], method, cap)
                        if method == "fixed_mixture":
                            assert result["adopted"] == old["adopted"] and result["pairs"] == old["episodes"], (directory, kind, scope, result, old["episodes"])
                        rows.append(dict(study=study, plant=directory.name, support=kind, scope=scope,
                                         method=method, bound=old["bound"], alpha=old["alpha"], cap=cap,
                                         raw_independent_gain=gain, deployed_independent_gain=gain if result["adopted"] else 0,
                                         training_transitions=old["training_transitions"],
                                         calibration_transitions=old.get("calibration_transitions", 0),
                                         total_consumed_transitions=old["training_transitions"] + old.get("calibration_transitions", 0) + 80*result["pairs"],
                                         **result))
    summary = []
    for key in sorted({(r["study"], r["scope"], r["method"], r["support"]) for r in rows}):
        group = [r for r in rows if (r["study"], r["scope"], r["method"], r["support"]) == key]
        summary.append(dict(zip(("study", "scope", "method", "support"), key),
                            plants=len(group), adoptions=sum(r["adopted"] for r in group),
                            rejections=sum(r["outcome"] == "reject" for r in group),
                            censored=sum(r["outcome"] == "inconclusive" for r in group),
                            mean_consumed_pairs=float(np.mean([r["pairs"] for r in group])),
                            mean_deployed_gain=float(np.mean([r["deployed_independent_gain"] for r in group]))))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, obj in [("records.json", rows), ("summary.json", summary), ("input_hashes.json", manifest)]:
        (OUT/name).write_text(json.dumps(obj, indent=2) + "\n")
    (OUT/"audit.json").write_text(json.dumps(dict(executed_utc=datetime.now(timezone.utc).isoformat(),
        records=len(rows), plants=64, original_mixture_replays=256, original_mixture_exact_matches=True,
        protocol_sha256=digest(protocol), status="retrospective robustness replay; no new training"), indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    return summary


def stochastic_example():
    # Exact analytic contrasts, rather than a noisy estimate of the zero variance.
    ds = np.geomspace(.005, .5, 25)
    rows = []
    for d in ds:
        mu = d * math.sqrt(1 - d*d/4)
        result = dict(displacement=float(d), gain=mu, branch_variance=1/3,
                      independent_contrast_variance=2/3, coupled_contrast_variance=0,
                      global_bound=4, paired_bound=float(d),
                      range_only_power_lower_bound=math.log(.95/.05)/math.log1p(mu/4))
        for method in ("fixed_mixture", "predictable_empirical_bernstein"):
            for kind, b in [("global", 4), ("paired", d)]:
                stopped = stop(np.full(100000, mu), b, .05, method, 100000, block=1)
                assert stopped["adopted"]
                result[method + "_" + kind] = stopped["pairs"]
        rows.append(result)
    (OUT/"stochastic_example.json").write_text(json.dumps(rows, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), layout="constrained")
    axes[0].semilogx(ds, np.full(len(ds), 2/3), color="#a65516", label="Independent difference")
    axes[0].semilogx(ds, np.full(len(ds), 1/3), color="#777777", ls="--", label="Each return")
    axes[0].semilogx(ds, np.zeros(len(ds)), color="#146b76", lw=2, label="Coupled difference")
    axes[0].set(xlabel="Action displacement d", ylabel="Population variance", ylim=(-.055, .8))
    for value, label, color in [(2/3, "Independent difference", "#a65516"),
                                (1/3, "Each return", "#777777"),
                                (0, "Coupled difference", "#146b76")]:
        axes[0].text(.006, value+.04, label, color=color, fontsize=8)
    for method, ls, label in [("fixed_mixture", "-", "Betting"), ("predictable_empirical_bernstein", "--", "PM-EB")]:
        for kind, color, name in [("global", "#a65516", "global"), ("paired", "#146b76", "paired")]:
            axes[1].loglog(ds, [r[method+"_"+kind] for r in rows], color=color, ls=ls, label=f"{label}, {name}")
    axes[1].loglog(ds, [r["range_only_power_lower_bound"] for r in rows], ":", color="black", label="Range-only lower bound")
    axes[1].set(xlabel="Action displacement d", ylabel="Pairs to certify")
    axes[1].legend(fontsize=6.4, frameon=False, loc="lower left", bbox_to_anchor=(.0, .09))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    (ROOT/"manuscript/figs_revision").mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT/f"manuscript/figs_revision/v2_zero_variance.{suffix}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    replay()
    stochastic_example()
