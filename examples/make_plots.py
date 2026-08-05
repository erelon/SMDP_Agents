"""Turn the runner's JSON into figures.

    python -m examples.make_plots                    # results/ -> results/plots/
    python -m examples.make_plots --env sincoslog
    python -m examples.make_plots --agents Harmonic SMART RLearning QLearning

Three kinds of figure:

``<env>_learning.png``
    Reward rate against elapsed time, one line per agent, averaged over seeds with
    a band for the spread. The rate is measured *per segment* rather than
    cumulatively, so the curve shows what the current policy earns instead of
    dragging the early transient along behind it — a converged agent flattens out,
    one still improving keeps climbing.
``<env>_rho.png``
    Each agent's own reward-rate estimate against time. On the ``criterion``
    environments this is the measurement: whether rho settles on the time-average
    or on the ratio of expectations is the whole question, and the two values are
    drawn as reference lines where they are known.
``family_<family>.png``
    One bar per agent per environment in the family, so the ranking across a whole
    family is visible at once. Agents whose actions the runner had to override are
    hatched and labelled, matching the report's ⚠.

Colours are assigned per agent and held fixed across every figure, so the same
line is the same agent everywhere.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

from .make_report import distorted, load, summarise
from .run import DEFAULT_RESULTS_DIR

DEFAULT_PLOT_DIR = os.path.join(DEFAULT_RESULTS_DIR, "plots")

#: Reference values worth drawing on the rho panels, per environment.
#: name -> ((value, label), ...)
RHO_REFERENCES: Dict[str, Tuple[Tuple[float, str], ...]] = {
    "feinberg": ((7.5, "time-average 7.5"), (10.0 / 1.5, "ratio of expectations 6.667")),
    "gemini": ((10.0, "time-average 10"), (1.0, "ratio of expectations 1"),
               (4.0, "the sure thing 4")),
    "ratio_vs_step_rate": ((1.99 / 990.01, "ratio of expectations 0.00201"),
                           (1.00099, "mean of rates 1.001")),
    "risk": ((25.0 / 3.0, "neutral 8.333"), (8.0, "seek / averse 8")),
    "two_states_uneven": ((0.693, "escape 0.693"), (0.667, "always-a 0.667")),
    "hell_or_heaven": ((0.998, "refuse 0.998"), (-0.798, "take -0.798")),
}


def palette(agents: Sequence[str]) -> Dict[str, tuple]:
    """A stable colour per agent, so a line means the same thing in every figure."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    return {name: cmap(i % 20) for i, name in enumerate(sorted(agents))}


def all_agents(records: Sequence[Dict]) -> List[str]:
    names = set()
    for record in records:
        names |= set(record["agents"])
    return sorted(names)


def segment_rates(curve: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    """Per-segment reward rate from cumulative checkpoints."""
    times, rates = [], []
    previous_t = previous_r = 0.0
    for point in curve:
        clock, reward = float(point[0]), float(point[1])
        span = clock - previous_t
        if span > 0:
            times.append(clock)
            rates.append((reward - previous_r) / span)
        previous_t, previous_r = clock, reward
    return times, rates


def rho_trace(curve: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    """Time and rho from the checkpoints, for runs recorded with the rho column."""
    times, rhos = [], []
    for point in curve:
        if len(point) < 3:
            continue
        times.append(float(point[0]))
        rhos.append(float(point[2]))
    return times, rhos


def _mean_curves(runs: Sequence[Dict], extract) -> Optional[Tuple[List, List, List]]:
    """Mean and spread of one extracted curve across seeds, on a shared grid."""
    import numpy as np

    series = [extract(run.get("curve") or []) for run in runs]
    series = [(t, v) for t, v in series if len(t) > 1]
    if not series:
        return None
    length = min(len(t) for t, _ in series)
    times = np.mean([t[:length] for t, _ in series], axis=0)
    values = np.array([v[:length] for _, v in series], dtype=float)
    finite = np.where(np.isfinite(values), values, np.nan)
    # An agent whose rho diverged on every seed leaves an all-NaN column, which
    # nanmean warns about; treat those points as zero rather than emit a warning
    # per figure. The report is where divergence is called out by name.
    usable = np.any(np.isfinite(finite), axis=0)
    mean = np.zeros(finite.shape[1])
    spread = np.zeros(finite.shape[1])
    if usable.any():
        # errstate: a rho that has run away to 1e200 overflows when squared for the
        # standard deviation. The overflow is the point, not a problem — let it come
        # out as inf and be visible on the plot.
        with np.errstate(all="ignore"):
            mean[usable] = np.nanmean(finite[:, usable], axis=0)
            spread[usable] = np.nanstd(finite[:, usable], axis=0)
    return list(times), list(mean), list(spread)


def line_figure(record: Dict, extract, ylabel: str, title: str, path: str,
                colours: Dict[str, tuple], agents: Optional[Sequence[str]] = None,
                references: Sequence[Tuple[float, str]] = (),
                logy: bool = False, drop_flat_zero: bool = False) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    chosen = [name for name in sorted(record["agents"])
              if agents is None or name in agents]
    fig, axis = plt.subplots(figsize=(10.5, 5))
    drawn = skipped = 0
    for name in chosen:
        curves = _mean_curves(record["agents"][name], extract)
        if curves is None:
            continue
        times, mean, spread = (np.array(part) for part in curves)
        if drop_flat_zero and not np.any(np.abs(mean) > 1e-12):
            # Q-learning and the bandits have no rho: drawing thirteen lines on
            # top of each other at zero hides the agents that do.
            skipped += 1
            continue
        style = "--" if distorted(name, record["agents"][name]) else "-"
        label = name + (" ⚠" if style == "--" else "")
        axis.plot(times, mean, style, color=colours.get(name), linewidth=1.5,
                  label=label)
        axis.fill_between(times, mean - spread, mean + spread,
                          color=colours.get(name), alpha=0.10, linewidth=0)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    if skipped:
        axis.plot([], [], " ", label=f"({skipped} agents with no rho omitted)")

    for value, text in references:
        axis.axhline(value, color="0.35", linestyle=":", linewidth=1.0)
        axis.annotate(text, xy=(0.005, value), xycoords=("axes fraction", "data"),
                      fontsize=7, color="0.25", va="bottom")

    axis.set_xlabel("elapsed time (environment time units)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if logy:
        axis.set_yscale("symlog")
    axis.grid(True, alpha=0.3)
    # Outside the axes: with thirteen agents an inset legend covers the data.
    axis.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5),
                frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def family_figure(records: Sequence[Dict], family: str, metric: str, path: str,
                  colours: Dict[str, tuple]) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    chosen = [r for r in records if r["family"] == family]
    if not chosen:
        return None
    agents = sorted(set().union(*(r["agents"] for r in chosen)))
    width = 0.8 / max(len(agents), 1)

    fig, axis = plt.subplots(figsize=(max(8, 1.7 * len(chosen) + 4), 5))
    for index, name in enumerate(agents):
        offsets, heights, errors, hatches = [], [], [], []
        for position, record in enumerate(chosen):
            runs = record["agents"].get(name)
            if not runs:
                continue
            mean, spread = summarise(runs, metric)
            offsets.append(position + index * width - 0.4 + width / 2)
            heights.append(mean)
            errors.append(spread)
            hatches.append("//" if distorted(name, runs) else "")
        if not offsets:
            continue
        bars = axis.bar(offsets, heights, width=width, yerr=errors, capsize=1.5,
                        color=colours.get(name), label=name, error_kw={"linewidth": 0.6})
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_edgecolor("0.2")

    axis.set_xticks(range(len(chosen)))
    axis.set_xticklabels([r["env"] for r in chosen], rotation=20, ha="right",
                         fontsize=8)
    axis.set_ylabel(metric)
    axis.set_title(f"{family}: {metric} by agent  (hatched = actions overridden)")
    axis.axhline(0.0, color="0.5", linewidth=0.8)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out", default=DEFAULT_PLOT_DIR)
    parser.add_argument("--env", action="append", help="restrict to these envs")
    parser.add_argument("--agents", nargs="+", help="restrict to these agents")
    parser.add_argument("--metric", default="lifetime_rate",
                        help="metric for the per-family bar charts")
    parser.add_argument("--no-families", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is not installed; nothing to do")
        return 1

    records = load(args.results)
    if args.env:
        records = [r for r in records if r["env"] in set(args.env)]
    if not records:
        print(f"no result files in {args.results}")
        return 1

    colours = palette(all_agents(records))
    written: List[str] = []
    for record in records:
        env = record["env"]
        note = record.get("note", "")
        path = line_figure(
            record, segment_rates, "reward rate over the segment",
            f"{env} — learning curve\n{note}",
            os.path.join(args.out, f"{env}_learning.png"), colours, args.agents)
        if path:
            written.append(path)
        path = line_figure(
            record, rho_trace, "the agent's rho estimate",
            f"{env} — rho over time\n{note}",
            os.path.join(args.out, f"{env}_rho.png"), colours, args.agents,
            references=RHO_REFERENCES.get(env, ()), drop_flat_zero=True)
        if path:
            written.append(path)

    if not args.no_families:
        for family in sorted({r["family"] for r in records}):
            path = family_figure(records, family, args.metric,
                                 os.path.join(args.out, f"family_{family}.png"),
                                 colours)
            if path:
                written.append(path)

    print(f"wrote {len(written)} figures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
