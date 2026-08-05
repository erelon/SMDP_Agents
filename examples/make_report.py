"""Turn the runner's JSON into a Markdown report.

    python -m examples.make_report                       # results/ -> results/REPORT.md
    python -m examples.make_report --metric greedy_rate
    python -m examples.make_report --results other/ --out other/REPORT.md

Produces, in order: a winner per environment, a win tally across agents, a
per-environment leaderboard, and a caveats section. Budgets and hyperparameters
come from each environment's source repository, and the leaderboards state which.

The caveats cover two things worth not glossing over: agents whose own rho
estimate diverged (their rates are still measured, but the estimate driving them
had stopped being a number), and any agent whose action choices the runner had to
override. The latter should now be empty — every agent routes its choice through
``get_available_actions`` — so a non-empty section there means an agent regressed.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import correct_actions
from .envs import FAMILIES
from .run import DEFAULT_RESULTS_DIR

#: An agent whose actions were overridden more often than this is not comparable.
ILLEGAL_TOLERANCE = 0.01

#: Agents for which the substitution is harmless. ``RandomAgent`` draws uniformly
#: from the full action space, so with the runner resampling its illegal draws
#: uniformly from the legal set the result is
#: ``1/|A| + (|A| - |L|)/|A| * 1/|L| = 1/|L|`` — exactly the uniform-over-legal
#: baseline it is meant to be. The bandits are different: their draw is an argmax
#: carrying information, and replacing it discards the thing being measured.
SUBSTITUTION_HARMLESS = frozenset({"RandomAgent"})

#: Agents with privileged information. They are the *ceiling*, not competitors —
#: ``Oracle`` is handed the environment's own optimal policy, so ranking it against
#: agents that had to learn one would just report that cheating works. Reported in
#: its own column instead. ``RandomAgent`` is deliberately not here: it is a
#: baseline but an honest one, and if it wins something that is worth knowing.
PRIVILEGED = frozenset({"Oracle"})

METRICS = ("lifetime_rate", "window_rate", "greedy_rate")


def load(results_dir: str) -> List[Dict[str, Any]]:
    """Every per-environment record in ``results_dir``, families in report order."""
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path).startswith("_"):
            continue
        with open(path) as handle:
            record = json.load(handle)
        if "agents" in record and "env" in record:
            records.append(record)
    order = {family: i for i, family in enumerate(FAMILIES)}
    records.sort(key=lambda r: (order.get(r.get("family"), len(FAMILIES)), r["env"]))
    return records


def summarise(runs: Sequence[Dict[str, Any]], key: str) -> Tuple[float, float]:
    """Mean and sample standard deviation of ``key`` over the finite seeds.

    Non-finite values are skipped rather than poisoning the mean: an agent whose
    rho diverges still has a measurable rate, and :func:`diverged` counts the
    seeds it happened on so the table can say so.
    """
    values = [float(run[key]) for run in runs if key in run]
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan"), 0.0
    spread = statistics.stdev(finite) if len(finite) > 1 else 0.0
    return statistics.fmean(finite), spread


#: A rho past this has stopped being an estimate of anything, whether or not it has
#: reached infinity yet. R-learning's rho update is not scaled by the holding time,
#: so on an environment with large tau the feedback gain exceeds 1 and it runs away —
#: which shows up as 1e200 as readily as as inf.
RUNAWAY = 1e6


def diverged(runs: Sequence[Dict[str, Any]], key: str) -> int:
    """How many seeds produced a non-finite or runaway ``key``."""
    return sum(1 for run in runs if key in run
               and not (math.isfinite(float(run[key]))
                        and abs(float(run[key])) <= RUNAWAY))


def overridden(runs: Sequence[Dict[str, Any]]) -> float:
    """The worst fraction of decisions the runner had to replace."""
    return max((float(run.get("illegal_frac", 0.0)) for run in runs), default=0.0)


def distorted(name: str, runs: Sequence[Dict[str, Any]]) -> bool:
    """Whether action substitution invalidated this agent's numbers."""
    return (name not in SUBSTITUTION_HARMLESS
            and overridden(runs) > ILLEGAL_TOLERANCE)


def comparable(record: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """The agents eligible to win: undistorted, and without privileged information."""
    return {name: runs for name, runs in record["agents"].items()
            if not distorted(name, runs) and name not in PRIVILEGED}


def ceiling(record: Dict[str, Any], metric: str) -> str:
    """The privileged agents' scores, as a reference for the winner column."""
    parts = []
    for name in sorted(PRIVILEGED & set(record["agents"])):
        mean, spread = summarise(record["agents"][name], metric)
        parts.append(f"{name} {fmt(mean, spread)}")
    return ", ".join(parts) or "—"


def trap_probe(env_name: str) -> Optional[str]:
    """The ``choices`` key whose correctness decides the winner, if any.

    Only the *traps* get one. Where the correct action is also the one that pays best
    immediately, correctness is uninformative — most agents reach 100% — and the rate
    is the right discriminator. Where it is not, ranking by rate names the agent that
    took the bait, which is the whole problem.
    """
    for state, choice in correct_actions.choices(env_name).items():
        if choice.is_trap:
            return f"choice_{state}"
    return None


def ranked(record: Dict, metric: str,
           agents: Optional[Dict[str, List[Dict]]] = None) -> List[Tuple]:
    """Agents ordered by correct-choice rate where that decides, then by ``metric``.

    Returns ``(name, correct_or_None, mean, spread)``, best first.
    """
    chosen = record["agents"] if agents is None else agents
    probe = trap_probe(record["env"])
    rows = []
    for name, runs in chosen.items():
        mean, spread = summarise(runs, metric)
        correct = None
        if probe is not None:
            rates = choice_rates(runs, probe, record["env"])
            correct = rates[0] if rates is not None else None
        rows.append((name, correct, mean, spread))
    rows.sort(key=lambda row: (row[1] if row[1] is not None else 0.0, row[2]),
              reverse=True)
    return rows


def fmt(value: float, spread: float = 0.0, places: int = 4) -> str:
    if value != value:  # NaN
        return "—"
    if abs(value) >= RUNAWAY or (value and abs(value) < 10 ** -places):
        return f"{value:.2e}" + (f" ± {spread:.1e}" if spread else "")
    if spread:
        return f"{value:.{places}f} ± {spread:.{places}f}"
    return f"{value:.{places}f}"


def winner_table(records: Sequence[Dict], metric: str) -> List[str]:
    lines = ["| Family | Environment | Budget | Seeds | "
             f"Winner ({metric}) | Runner-up | Ceiling |",
             "|---|---|--:|--:|---|---|---|"]

    def budget_of(record):
        unit = record.get("budget_unit", "time")
        return f"{record['budget']:,.0f} {'steps' if unit == 'steps' else 'time'}"
    for record in records:
        rows = ranked(record, metric, comparable(record))
        trap = trap_probe(record["env"]) is not None
        if not rows:
            lines.append(f"| {record['family']} | `{record['env']}` | "
                         f"{budget_of(record)} | {len(record['seeds'])} | "
                         f"— (every agent's actions were overridden) | — | "
                         f"{ceiling(record, metric)} |")
            continue

        def label(row):
            name, correct, mean, spread = row
            text = f"{name} {fmt(mean, spread)}"
            if correct is not None:
                text = f"{name} {correct:.0%} correct, {fmt(mean, spread)}"
            return text

        top = rows[0]
        best = "**" + label(top).split(" ", 1)[0] + "** " + label(top).split(" ", 1)[1]
        # Ties: on a trap, only agents matching the leader's correctness compete, and
        # among those the metric decides within one standard deviation. Off a trap the
        # metric alone decides. On the criterion environments the tie is deliberately
        # everyone, and naming one of them a winner would be reading noise.
        peers = [row for row in rows if row[1] == top[1]]
        margin = max(top[3], abs(top[2]) * 1e-9)
        tied = sum(1 for row in peers if abs(top[2] - row[2]) <= margin)
        if tied >= 3:
            share = f"{top[1]:.0%} correct, " if top[1] is not None else ""
            best = (f"*{tied}-way tie* at {share}{fmt(top[2], top[3])}"
                    + (" — every agent" if tied == len(rows) else ""))
            second = "—"
        else:
            second = label(rows[1]) if len(rows) > 1 else "—"
            if tied == 2:
                best += " *(tied with the runner-up)*"
        lines.append(f"| {record['family']} | `{record['env']}` | "
                     f"{budget_of(record)} | {len(record['seeds'])} | "
                     f"{best} | {second} | {ceiling(record, metric)} |")
    return lines


def tally_table(records: Sequence[Dict], metric: str) -> List[str]:
    """Wins per agent, skipping the environments where three or more agents tie."""
    tally: Dict[str, int] = {}
    for record in records:
        rows = ranked(record, metric, comparable(record))
        if not rows:
            continue
        top = rows[0]
        peers = [row for row in rows if row[1] == top[1]]
        margin = max(top[3], abs(top[2]) * 1e-9)
        if sum(1 for row in peers if abs(top[2] - row[2]) <= margin) >= 3:
            continue
        tally[top[0]] = tally.get(top[0], 0) + 1
    if not tally:
        return ["No comparable results."]
    lines = ["| Agent | Wins |", "|---|--:|"]
    for name, wins in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {name} | {wins} |")
    return lines


def leaderboard(record: Dict, metric: str) -> List[str]:
    lines = [f"### `{record['env']}`  ({record['family']})", "",
             record.get("note", ""), ""]
    episodes = summarise(next(iter(record["agents"].values())), "resets")[0]
    unit = "decisions" if record.get("budget_unit") == "steps" else "time units"
    lines.append(f"Budget {record['budget']:,.0f} {unit}, "
                 f"{len(record['seeds'])} seeds, warmup {record['warmup_frac']:.0%}"
                 + (f", ~{episodes:,.0f} restarts per run" if episodes else "")
                 + f". Sorted by {metric}.")
    if record.get("source"):
        note = f"Protocol and hyperparameters from `{record['source']}`"
        if record.get("attributed") is False:
            note += " — **the source pins no budget for this environment**, so the "\
                    "budget above is this repo's choice, not provenance"
        if record.get("source_seeds"):
            note += f". The source itself used {record['source_seeds']} seed(s)"
        greedy = record.get("greedy_budget")
        if greedy:
            origin = ("the source's own" if record.get("greedy_from_source")
                      else "**not from the source**, which ran no greedy evaluation")
            note += f". Greedy budget {greedy:,.0f} ({origin})"
        schedule = record.get("epsilon_schedule")
        if schedule:
            note += (f". Epsilon decays {schedule['start']} -> {schedule['end']} "
                     f"linearly over {schedule['decay_fraction']:.0%} of the run")
        lines += ["", note + "."]
    used = {name: hp for name, hp in (record.get("hyperparameters") or {}).items() if hp}
    if used:
        lines += ["", "<details><summary>Hyperparameters</summary>", ""]
        lines += ["| Agent | Settings |", "|---|---|"]
        for name in sorted(used):
            settings = ", ".join(f"`{k}={v}`" for k, v in sorted(used[name].items()))
            lines.append(f"| {name} | {settings} |")
        missing = sorted(set(record["agents"]) - set(used))
        if missing:
            lines += ["", f"Library defaults: {', '.join(missing)}."]
        lines += ["", "</details>"]
    lines += ["",
              "| Agent | lifetime_rate | window_rate | greedy_rate | rho | states | overridden |",
              "|---|--:|--:|--:|--:|--:|--:|"]
    for name, _, _, _ in ranked(record, metric):
        runs = record["agents"][name]
        share = overridden(runs)
        flag = " ⚠" if distorted(name, runs) else ""
        cells = [f"{name}{flag}"]
        for key in ("lifetime_rate", "window_rate", "greedy_rate"):
            mean, spread = summarise(runs, key)
            cells.append(fmt(mean, spread))
        rho = fmt(summarise(runs, "rho")[0], places=3)
        blown = diverged(runs, "rho")
        cells.append(f"{rho} (diverged {blown}/{len(runs)})" if blown else rho)
        cells.append(f"{summarise(runs, 'states')[0]:,.0f}")
        cells.append(f"{share:.0%}" if share else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines += choice_table(record, metric)
    lines.append("")
    return lines


def divergence_notes(records: Sequence[Dict]) -> List[str]:
    """List every (agent, environment) whose rho estimate went non-finite."""
    blown: List[Tuple[str, str, int, int]] = []
    for record in records:
        for name, runs in record["agents"].items():
            count = diverged(runs, "rho")
            if count:
                blown.append((name, record["env"], count, len(runs)))
    if not blown:
        return []
    lines = [
        "### Diverged rho estimates",
        "",
        f"These agents' own reward-rate estimates went non-finite or ran past "
        f"{RUNAWAY:.0e}. The rates reported for them are still measured from the "
        f"trajectory, but the estimate driving their updates had stopped being an "
        f"estimate of anything.",
        "",
        "The pattern to look for is a rho update that is **not scaled by the "
        "holding time** meeting an environment whose holding times are large: "
        "R-learning's update is `rho += beta * td_error` with `td_error` "
        "containing `-rho * tau`, so at `beta = 0.03` and `tau = 500` the feedback "
        "gain is far above 1 and rho runs away. The discrete `RLearning` is "
        "unaffected because it clamps `tau` to 1.",
        "",
        "| Agent | Environment | Seeds affected |",
        "|---|---|--:|",
    ]
    for name, env, count, total in sorted(blown):
        lines.append(f"| {name} | `{env}` | {count}/{total} |")
    return lines + [""]


def caveats(records: Sequence[Dict]) -> List[str]:
    offenders: Dict[str, List[Tuple[str, float, bool]]] = {}
    for record in records:
        for name, runs in record["agents"].items():
            share = overridden(runs)
            if share > ILLEGAL_TOLERANCE:
                offenders.setdefault(name, []).append(
                    (record["env"], share, distorted(name, runs)))
    if not offenders:
        return ["Every agent chose only legal actions; nothing was overridden."]
    lines = [
        "### Overridden actions",
        "",
        "**This section should be empty.** Every agent chooses through "
        "`get_available_actions`, so none should ever propose an action the "
        "environment refuses; where one does, the runner substitutes a uniformly "
        "random legal action and counts it here. A non-empty table means an agent "
        "regressed.",
        "",
        "For a bandit a substitution is fatal — the draw being replaced was an "
        "argmax carrying what the agent had learned — so it is excluded from the "
        "winner columns (marked ⚠). `RandomAgent` is not excluded: drawing "
        "uniformly from all `|A|` actions and resampling the illegal ones "
        "uniformly from the `|L|` legal ones gives each legal action "
        "`1/|A| + (|A|-|L|)/|A| · 1/|L| = 1/|L|`, exactly the uniform-over-legal "
        "baseline it is meant to be.",
        "",
        "| Agent | Environment | Decisions overridden | Excluded |",
        "|---|---|--:|---|",
    ]
    for name in sorted(offenders):
        for env, share, excluded in sorted(offenders[name], key=lambda row: -row[1]):
            lines.append(f"| {name} | `{env}` | {share:.0%} | "
                         f"{'yes' if excluded else 'no — equivalent'} |")
    return lines


def choice_rates(runs: Sequence[Dict[str, Any]], state: str,
                 env_name: Optional[str] = None) -> Optional[Tuple]:
    """``(correct_fraction, bait_fraction, n)`` for one probe across seeds.

    Correctness is computed here, against the table as it stands now, from the action
    the agent was recorded as choosing. Older result files that stored a verdict are
    read for their ``chosen`` field only.
    """
    chosen = []
    for run in runs:
        entry = run.get("choices", {}).get(state)
        if entry is None:
            continue
        chosen.append(entry["chosen"] if isinstance(entry, dict) else entry)
    if not chosen:
        return None
    name = env_name or (runs[0].get("env") if runs else None)
    choice = correct_actions.choices(name).get(_state_of_key(state)) if name else None
    if choice is None:
        return None
    correct_action = choice.resolve(make_probe(name))
    n = len(chosen)
    correct = sum(1 for a in chosen if a == correct_action) / n
    bait = sum(1 for a in chosen if choice.bait is not None and a == choice.bait) / n
    return correct, bait, n


def _state_of_key(key: str):
    """``choice_s1`` -> ``"s1"``, ``choice_0`` -> ``0``."""
    raw = key[len("choice_"):]
    try:
        return int(raw)
    except ValueError:
        return raw


def choice_table(record: Dict, metric: str) -> List[str]:
    """Per-agent correct-choice rates, for the environments that define one."""
    probes = correct_actions.choices(record["env"])
    if not probes:
        return []
    lines: List[str] = []
    for state, choice in probes.items():
        key = f"choice_{state}"
        rows = []
        for name, runs in record["agents"].items():
            if name in PRIVILEGED:
                continue
            rates = choice_rates(runs, key, record["env"])
            if rates is not None:
                rows.append((rates[0], -rates[1], name, rates))
        if not rows:
            continue
        rows.sort(reverse=True)
        kind = "trap" if choice.is_trap else "no bait"
        env = make_probe(record["env"])
        detail = choice.detail(env) if choice.detail else ""
        lines += ["",
                  f"**Correct choice at `{state}`** — the correct action is "
                  f"`{choice.resolve(env)}` under *{choice.criterion}*"
                  + (f", against a bait of `{choice.bait}` which "
                     f"{choice.bait_by}" if choice.is_trap else "")
                  + f" ({kind}). {choice.note}."
                  + (f" {detail[0].upper() + detail[1:]}." if detail else ""),
                  "",
                  "| Agent | Chose correctly | Took the bait |",
                  "|---|--:|--:|"]
        for _, _, name, (correct, bait, n) in rows:
            baited = f"{bait:.0%}" if choice.is_trap else "—"
            lines.append(f"| {name} | {correct:.0%} of {n} | {baited} |")
    return lines


_probe_envs: Dict[str, Any] = {}


def make_probe(env_name: str):
    """A built environment, cached, only so a callable correct action can resolve."""
    if env_name not in _probe_envs:
        from .envs import make
        _probe_envs[env_name] = make(env_name)
    return _probe_envs[env_name]


def trap_summary(records: Sequence[Dict]) -> List[str]:
    """One table: on every trap, how often each agent got the answer right."""
    trap_records = [r for r in records
                    if any(c.is_trap for c in correct_actions.choices(r["env"]).values())]
    if not trap_records:
        return []
    # Oracle is excluded for the same reason it is excluded from the winner column:
    # it is handed the environment's own optimal policy, and on the latent-phase
    # environments it is not even a stationary policy, so scoring it against a
    # single correct action measures nothing.
    agents = sorted(set().union(*(r["agents"] for r in trap_records)) - PRIVILEGED)
    header = ["| Agent | " + " | ".join(f"`{r['env']}`" for r in trap_records)
              + " | Correct |", "|---|" + "--:|" * (len(trap_records) + 1)]
    body = []
    for name in agents:
        cells, hits, total = [], 0, 0
        for record in trap_records:
            state = next(s for s, c in correct_actions.choices(record["env"]).items()
                         if c.is_trap)
            rates = choice_rates(record["agents"].get(name, []), f"choice_{state}",
                                 record["env"])
            if rates is None:
                cells.append("—")
                continue
            cells.append(f"{rates[0]:.0%}")
            hits += rates[0] * rates[2]
            total += rates[2]
        share = hits / total if total else 0.0
        body.append((share, f"| {name} | " + " | ".join(cells)
                     + f" | **{share:.0%}** |"))
    body.sort(reverse=True)
    return ["## Correct choices on the trap environments",
            "",
            "On these environments the correct action is *not* the one that looks "
            "best while the agent is learning, so a high reward rate can mean an "
            "agent took the bait and was paid for it. This table is the measurement "
            "the environments' own source uses instead: the greedy action at the "
            "decision state after training, scored against a known-correct one. "
            "Percentages are over seeds; see each leaderboard for the criterion each "
            "correct action is correct under.",
            "",
            *header, *(line for _, line in body), ""]


def build(records: Sequence[Dict], metric: str) -> str:
    if not records:
        return "# Example environments — results\n\nNo result files found.\n"
    seeds = sorted({len(r["seeds"]) for r in records})
    parts = [
        "# Example environments — results",
        "",
        f"{len(records)} environments, "
        f"{len(set().union(*(r['agents'] for r in records)))} agents, "
        f"{'/'.join(str(s) for s in seeds)} seeds. "
        f"Ranked by `{metric}`.",
        "",
        "All rates are reward per unit time. `lifetime_rate` covers the whole "
        "run and so charges an agent for its learning transient and its permanent "
        "ε-exploration; `window_rate` covers the post-warmup tail; `greedy_rate` "
        "is a separate run with exploration and learning switched off. `rho` is "
        "the agent's own final reward-rate estimate, which is the measurement of "
        "interest on the `criterion` environments.",
        "",
        "`Oracle` is handed the environment's own optimal policy, so it is reported "
        "as the **ceiling** rather than ranked against agents that had to learn "
        "one. It still appears in the per-environment tables.",
        "",
        "Every budget and every hyperparameter comes from the environment's source "
        "repository — nothing here is tuned by this repo. The budget *unit* varies "
        "because the sources disagree: whack-a-mole is bounded by simulated time, "
        "which is the fair unit when holding times vary, and the rest by decisions. "
        "See `examples/source_settings.py`.",
        "",
        "Reproduce with `python -m examples.run --all --seeds "
        f"{seeds[-1]}` then `python -m examples.make_report`.",
        "",
        "## Winner per environment",
        "",
        "On the **trap** environments — where the correct action is not the one that "
        "looks best while the agent is learning — the winner is decided by the "
        "correct-choice rate first and the metric only among agents that match it. "
        "Ranking those by rate alone would name whichever agent took the bait and "
        "got paid for it. Everywhere else the metric decides. Cells showing "
        "`N% correct` are the trap environments; see "
        "[Correct choices](#correct-choices-on-the-trap-environments).",
        "",
        *winner_table(records, metric),
        "",
        "## Win tally",
        "",
        *tally_table(records, metric),
        "",
    ]
    parts += trap_summary(records)
    parts += ["## Per-environment leaderboards", ""]
    for record in records:
        parts += leaderboard(record, metric)
    parts += ["## Caveats", "", *divergence_notes(records), *caveats(records), ""]
    return "\n".join(parts)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=DEFAULT_RESULTS_DIR,
                        help="directory of per-environment JSON")
    parser.add_argument("--out", help="output path (default <results>/REPORT.md)")
    parser.add_argument("--metric", default="lifetime_rate", choices=METRICS,
                        help="metric to rank by")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    records = load(args.results)
    out = args.out or os.path.join(args.results, "REPORT.md")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(build(records, args.metric))
    print(f"wrote {out} from {len(records)} result files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
