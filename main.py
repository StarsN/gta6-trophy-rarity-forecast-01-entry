"""gta6-trophy-rarity-forecast-01: forecast GTA VI's PS5 trophy earn rates on Exophase.

The program reads one case (makefun/gta6-trophies-case/v1): the trophy list and every read the
house made of the game's Exophase page before the case's checkpoint. For each trophy it writes
five quantiles (the case's levels) of the earn rate — the percent of Exophase-tracked players who
will hold the trophy at the case's target — scored by pinball loss on the logit scale.

Model (all of it runs offline, standard library only, deterministic):

  1. The history is a step function: Exophase refreshes its counts on its own clock, so the
     recorder's reads repeat one page state many times. The reads are compressed to the distinct
     page states, and states from before the game's release (a small pre-release population of
     reviewers, whose rates sit on a different trajectory) are set aside.
  2. Each trophy is classified from its name/description/tier (story chapter, story finale,
     named mission, stranger strand, grind, misc, 100%-completion, platinum, saturated intro).
     The class carries a prior for how fast the rate's logit grows per unit of log-time, a
     ceiling the median may not cross, and a spread floor.
  3. Per trophy, the forecast is a growth of the last observed rate on the logit scale. The
     growth mixes three estimates: the trophy's own weighted least-squares slope of
     logit(rate) against u = ln(1 + hours since release / 12) over the distinct post-release
     states (weighted by sqrt(players tracked), shrunk toward the class prior, winsorised so
     one lumpy Exophase refresh cannot dominate); the pooled growth of the trophies sitting
     at a similar rate level (level-bucket growth, damped); and the class prior slope. The
     total growth is also capped, so no trophy's odds multiply without limit, and floored,
     so a checkpoint dip is not extrapolated into a crash.
  4. The story chain (Chapter 1..4, then the final mission) also gets a pass-through
     forecast: each chapter's rate at the target as a fraction of the previous chapter's —
     the fraction rises down the chain, because the players who get far are the committed
     ones — blended with a catch-up dynamic measured from the current earner ratio when
     that ratio is measurable. The chain's medians are then made non-increasing.
  5. Trophies with no earners yet in the tracked population (the platinum, the 100% trophy and
     the longest grinds on day two) have no slope to fit: their median comes from the class
     prior path, scaled by the days since release at the target, with a wide spread.
  6. Spreads combine a base term, the fit's residual scale grown by the extrapolation
     distance, and a class term, floored so no trophy is forecast with false confidence.
  7. If a case has no usable post-release states at all (the practice shape: only the stale
     pre-release page), a dilution fallback applies: the tracked population multiplies over
     the horizon while earners grow slowly, so rates fall, with a wide spread.

Quantiles are sigmoid(median_logit + z_tau * sigma), rounded to 4 decimals (half up), clamped
to [0.01, 99.99] and made non-decreasing. Runtime is milliseconds per case.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

# The grader runs `python3 -P -s /bot/main.py`, which leaves this file's directory off
# sys.path; the program is a single file and needs no sibling modules, but keep the path
# line so local runs and the grader behave the same.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

TAU_HOURS = 12.0          # log-time scale: u = ln(1 + hours_since_release / TAU_HOURS)
BASE_SIGMA = 0.30         # spread floor component every trophy carries
SHRINK_K = 3.0            # slope shrinkage strength toward the class prior
SLOPE_BOOST = 1.08        # mild steepening: dilution fades as tracked growth slows

# Per-class priors: slope of logit(rate) per unit u, the slope cap, ceiling for the median
# (percent), spread component, and the day-7 median (percent) used when a trophy has no
# earners yet.
CLASS = {
    "story":       {"slope": 1.05, "scap": 1.5, "cap": 48.0, "sigma": 0.35, "zero7": 0.05},
    "story_final": {"slope": 1.15, "scap": 1.5, "cap": 38.0, "sigma": 0.38, "zero7": 0.04},
    "mission":     {"slope": 0.95, "scap": 1.4, "cap": 40.0, "sigma": 0.40, "zero7": 0.06},
    "strand":      {"slope": 0.90, "scap": 1.4, "cap": 30.0, "sigma": 0.45, "zero7": 0.05},
    "misc":        {"slope": 0.85, "scap": 1.3, "cap": 40.0, "sigma": 0.45, "zero7": 0.05},
    "grind":       {"slope": 0.85, "scap": 1.4, "cap": 20.0, "sigma": 0.55, "zero7": 0.08},
    "completion":  {"slope": 1.10, "scap": 1.6, "cap": 6.0,  "sigma": 0.70, "zero7": 0.10},
    "platinum":    {"slope": 1.20, "scap": 1.8, "cap": 4.0,  "sigma": 0.80, "zero7": 0.06},
}

# Pass-through priors for the story chain at the target: the rate of each chapter as a
# fraction of the previous chapter's rate (later chapters keep a larger share — the players
# who get that far are the committed ones).
CHAIN_PHI = [0.62, 0.72, 0.78, 0.80]

GRIND_WORDS = (
    "stunt", "gold medal", "every location", "every spot", "card set", "challenges",
    "all shooting", "all garage", "all of", "every ", "100 locations", "250 ",
)


def parse_time(text):
    """ISO 8601 UTC text -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int (JSON-compact)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def u_of(hours_since_release):
    return math.log1p(max(hours_since_release, 0.0) / TAU_HOURS)


def classify(trophy):
    """Return (class_name, story_order or None)."""
    desc = trophy.get("description", "") or ""
    low = desc.lower()
    tier = trophy.get("tier", "")
    if tier == "platinum":
        return "platinum", None
    if "100% completion" in low:
        return "completion", None
    m = re.search(r"complete chapter\s+(\d+)", low)
    if m:
        return "story", int(m.group(1))
    if low.startswith("complete the final mission."):
        return "story_final", 99
    if low.startswith("complete the final mission for"):
        return "strand", None
    if "mission strands for strangers" in low:
        return "strand", None
    if any(w in low for w in GRIND_WORDS):
        return "grind", None
    if low.startswith("complete "):
        return "mission", None
    return "misc", None


class State:
    __slots__ = ("t_hours", "players", "rates")

    def __init__(self, t_hours, players, rates):
        self.t_hours = t_hours
        self.players = players
        self.rates = rates


def distinct_states(history, release):
    """Compress reads to distinct page states (Exophase refresh steps), oldest first."""
    states = []
    for read in history:
        players = read.get("players_tracked") or 0
        rates = list(read.get("rates") or [])
        if states and states[-1].players == players and states[-1].rates == rates:
            continue
        t = (parse_time(read["at"]) - release).total_seconds() / 3600.0
        states.append(State(t, players, rates))
    return states


def wls_line(xs, ys, ws):
    """Weighted least squares y = ybar + b (x - xbar); returns (xbar, ybar, slope, resid_sd)."""
    sw = sum(ws)
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return xbar, ybar, 0.0, 0.0
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    resid = [y - (ybar + slope * (x - xbar)) for x, y in zip(xs, ys)]
    var = sum(w * r * r for r, w in zip(resid, ws)) / sw
    return xbar, ybar, slope, math.sqrt(var)


def forecast_players(states, u_star):
    """Damped log-linear forecast of players tracked at u_star (growth factor clamped)."""
    post = [s for s in states if s.t_hours >= 0 and s.players > 0]
    if len(post) < 2:
        return None
    xs = [u_of(s.t_hours) for s in post]
    ys = [math.log(s.players) for s in post]
    ws = [1.0] * len(post)
    xbar, ybar, slope, _ = wls_line(xs, ys, ws)
    last = post[-1]
    ln_last = math.log(last.players)
    ln_star = ln_last + 0.55 * slope * (u_star - u_of(last.t_hours))
    factor = math.exp(ln_star - ln_last)
    factor = min(6.0, max(1.0, factor))
    return last.players * factor


def prior_median(cls, days_at_target):
    """Median (percent) for a trophy with no earners yet, from its class path."""
    base = CLASS[cls]["zero7"]
    days = max(days_at_target, 0.25)
    return base * (days / 7.0) ** 1.3


def fallback_forecast(case, trophies, last_rates, horizon_minutes, nd):
    """No usable post-release states: the tracked population explodes while earners crawl.

    Rates fall by dilution: median = last rate * earner-growth / population-growth, wide spread.
    """
    h_hours = horizon_minutes / 60.0
    if h_hours <= 3.0:
        g_pop = 6.0
    else:
        g_pop = min(30.0, 6.0 * (h_hours / 3.0) ** 0.25)
    h_days = horizon_minutes / 1440.0
    earner_growth = {
        "story": 1.0 + 6.0 * h_days, "story_final": 1.0 + 5.0 * h_days,
        "mission": 1.0 + 6.0 * h_days, "strand": 1.0 + 5.0 * h_days,
        "misc": 1.0 + 7.0 * h_days, "grind": 1.0 + 4.0 * h_days,
        "completion": 1.0 + 3.0 * h_days, "platinum": 1.0 + 3.0 * h_days,
    }
    z = [nd.inv_cdf(tau) for tau in case["levels"]]
    rows = []
    for i, trophy in enumerate(trophies):
        cls, _ = classify(trophy)
        r = last_rates[i]
        if r is None:
            r = 0.0
        if r >= 99.0:
            med, sigma = 99.97, 0.45
        else:
            med = clamp_rate(r * earner_growth[cls] / g_pop)
            med = min(med, CLASS[cls]["cap"])
            sigma = 1.25
        centre = logit(med)
        q = [round4(clamp_rate(sigmoid(centre + zz * sigma))) for zz in z]
        rows.append((trophy["id"], q))
    return rows


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    game = case.get("game") or {}
    rel_date = game.get("release_date") or "2026-11-19"
    release = datetime.fromisoformat(rel_date).replace(tzinfo=timezone.utc)
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_minutes = float(case["horizon_minutes"])
    nd = NormalDist()
    z = [nd.inv_cdf(tau) for tau in case["levels"]]

    states = distinct_states(case["history"], release)
    if not states:
        raise ValueError("case has no history")
    last = states[-1]
    last_rates = []
    for i in range(n):
        r = None
        for s in reversed(states):
            if i < len(s.rates) and s.rates[i] is not None:
                r = s.rates[i]
                break
        last_rates.append(r)

    post = [s for s in states if s.t_hours >= 0]
    # Usable launch data: at least two distinct post-release states and real population growth.
    usable = len(post) >= 2 and post[-1].players >= 2 * max(1, post[0].players if post else 1)
    if not usable:
        return fallback_forecast(case, trophies, last_rates, horizon_minutes, nd)

    u_star = u_of((target - release).total_seconds() / 3600.0)
    u_last = u_of(max(last.t_hours, 0.0))
    days_at_target = (target - release).total_seconds() / 86400.0
    horizon_days = horizon_minutes / 1440.0

    classes = [classify(t) for t in trophies]
    medians = [None] * n
    sigmas = [None] * n
    earners = [0.0] * n
    series = []
    for i in range(n):
        xs, ys, ws = [], [], []
        for s in post:
            if i < len(s.rates) and s.rates[i] is not None:
                xs.append(u_of(s.t_hours))
                ys.append(logit(s.rates[i]))
                ws.append(math.sqrt(max(s.players, 1) / max(post[-1].players, 1)))
        series.append((xs, ys, ws))
        if post and i < len(post[-1].rates) and post[-1].rates[i] is not None:
            earners[i] = post[-1].rates[i] * post[-1].players / 100.0
        elif last_rates[i] is not None:
            earners[i] = last_rates[i] * post[-1].players / 100.0

    # Level-bucket growth: how fast trophies at each current rate level grew, per unit u,
    # over the later part of the observed window (from the state nearest 40% of the window
    # to the last state — the early window still carries the launch-night refresh shock),
    # median within the bucket, clamped so the pool moderates rather than amplifies.
    u_first = u_of(max(post[0].t_hours, 0.0))
    du_obs = max(u_last - u_first, 0.05)
    ref = min(post, key=lambda s: abs(u_of(s.t_hours) - (u_first + 0.4 * du_obs)))
    du_ref = max(u_last - u_of(ref.t_hours), 0.05)
    bucket_edges = [-6.0, -4.0, -2.0]
    growths = {b: [] for b in range(len(bucket_edges) + 1)}
    all_growth = []
    for i in range(n):
        r0 = ref.rates[i] if i < len(ref.rates) else None
        r1 = post[-1].rates[i] if i < len(post[-1].rates) else None
        if r0 is None or r1 is None or r0 <= 0 or r1 <= 0:
            continue
        if max(r0 * ref.players, r1 * post[-1].players) / 100.0 < 1.0:
            continue
        g = (logit(r1) - logit(r0)) / du_ref
        lg = logit(r1)
        b = sum(1 for e in bucket_edges if lg > e)
        growths[b].append(g)
        all_growth.append(g)

    def bucket_growth(lg):
        b = sum(1 for e in bucket_edges if lg > e)
        vals = growths.get(b) or []
        if len(vals) < 3:
            vals = all_growth
        if not vals:
            return None
        vals = sorted(vals)
        return min(max(vals[len(vals) // 2], -1.0), 1.6)

    du_rem = u_star - u_last
    growth_cap = min(2.6, 1.35 * du_rem + 0.9)

    for i in range(n):
        cls, _order = classes[i]
        prior = CLASS[cls]
        r_last = last_rates[i]
        xs, ys, ws = series[i]

        if r_last is not None and r_last >= 99.0:
            # Saturated (the opening mission): printed 100.00 in every era so far.
            medians[i] = 99.98
            sigmas[i] = 0.45
            continue

        if r_last is None or r_last <= 0.0 or earners[i] < 1.0:
            # No earners yet in the tracked population: class prior path, wide spread.
            medians[i] = min(prior_median(cls, days_at_target), prior["cap"])
            sigmas[i] = 1.05
            continue

        centre0 = logit(r_last)
        # (a) The trophy's own slope, winsorised and shrunk toward the class prior.
        d_own = None
        sig_fit = 0.35
        if len(xs) >= 2:
            _xbar, _ybar, slope, resid = wls_line(xs, ys, ws)
            states_with = sum(1 for s in post
                              if i < len(s.rates) and s.rates[i] is not None
                              and s.rates[i] * s.players / 100.0 >= 3.0)
            info = min(states_with, 6) + min(earners[i] / 20.0, 2.0)
            slope = min(max(slope, -0.5), prior["scap"])
            slope_shrunk = (slope * info + prior["slope"] * SHRINK_K) / (info + SHRINK_K)
            d_own = slope_shrunk * SLOPE_BOOST * du_rem
            u_range = max(xs) - min(xs)
            lev = du_rem / (u_range + 0.3)
            sig_fit = min(max(resid, 0.20), 1.0) * (0.4 + 0.6 * min(lev, 2.0))
        # (b) Pooled growth of trophies at a similar level, damped.
        g_b = bucket_growth(centre0)
        d_bucket = g_b * du_rem * 0.85 if g_b is not None else None
        # (c) The class prior slope.
        d_prior = prior["slope"] * du_rem

        parts = []
        if d_own is not None:
            w_own = 0.55 if earners[i] >= 20 else 0.35
            parts.append((d_own, w_own))
        if d_bucket is not None:
            parts.append((d_bucket, 0.20 if earners[i] >= 20 else 0.30))
        parts.append((d_prior, 0.25))
        wsum = sum(w for _d, w in parts)
        delta = sum(d * w for d, w in parts) / wsum
        centre = centre0 + min(max(delta, -1.1), growth_cap)
        sigma = math.sqrt(BASE_SIGMA ** 2 + sig_fit ** 2 + prior["sigma"] ** 2)
        if earners[i] <= 2.0:
            pm = prior_median(cls, days_at_target)
            w = earners[i] / (earners[i] + 3.0)
            centre = w * centre + (1.0 - w) * logit(pm)
            sigma = min(max(sigma, 1.0), 1.25)
        medians[i] = min(sigmoid(centre), prior["cap"])
        sigmas[i] = min(max(sigma, 0.55), 1.6)

    # --- Story pass-through: each chapter as a fraction of the previous chapter. ---
    chain = sorted(
        (order, i) for i, (cls, order) in enumerate(classes)
        if cls in ("story", "story_final") and order is not None
    )
    if len(chain) >= 2:
        last_players = post[-1].players
        for pos in range(1, len(chain)):
            idx = chain[pos][1]
            prev_idx = chain[pos - 1][1]
            phi_prior = CHAIN_PHI[min(pos - 1, len(CHAIN_PHI) - 1)]
            e_now = (last_rates[idx] or 0.0) * last_players / 100.0
            e_prev_now = (last_rates[prev_idx] or 0.0) * last_players / 100.0
            phi_used = phi_prior
            if e_prev_now >= 10.0 and e_now >= 3.0:
                phi = min(e_now / e_prev_now, 0.98)
                phi_dyn = min(max(1.0 - (1.0 - phi) * math.exp(-horizon_days / 3.5),
                                  0.25), 0.92)
                phi_used = math.sqrt(phi_prior * phi_dyn)
            rate_fun = medians[prev_idx] * phi_used
            if earners[idx] >= 20:
                w_fun = 0.50
            elif earners[idx] >= 3:
                w_fun = 0.65
            else:
                w_fun = 0.80
            centre = (1.0 - w_fun) * logit(medians[idx]) + w_fun * logit(rate_fun)
            medians[idx] = min(sigmoid(centre), CLASS[classes[idx][0]]["cap"])
        # Medians must not increase along the story.
        run_min = None
        for _order, idx in chain:
            if run_min is not None and medians[idx] > run_min:
                medians[idx] = run_min
            run_min = medians[idx] if run_min is None else min(run_min, medians[idx])
        # Thin-data story content gets a structural floor from the finale's forecast:
        # a named story mission cannot sit far below the rate of finishing the story,
        # since the players who finish pass the story's missions on the way.
        final_idx = chain[-1][1]
        for i in range(n):
            if earners[i] >= 20:
                continue
            if classes[i][0] == "mission":
                medians[i] = max(medians[i], 0.8 * medians[final_idx])
            elif classes[i][0] == "strand":
                medians[i] = max(medians[i], 0.3 * medians[final_idx])

    # The platinum is earned only by players who also hold the 100% trophy.
    plat = [i for i, (c, _o) in enumerate(classes) if c == "platinum"]
    comp = [i for i, (c, _o) in enumerate(classes) if c == "completion"]
    if plat and comp and medians[plat[0]] > medians[comp[0]]:
        medians[plat[0]] = medians[comp[0]]

    rows = []
    for i, trophy in enumerate(trophies):
        centre = logit(medians[i])
        q = [round4(clamp_rate(sigmoid(centre + zz * sigmas[i]))) for zz in z]
        for k in range(1, len(q)):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append((trophy["id"], q))
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        rows = forecast_case(case)
        out = {"trophies": [{"id": tid, "q": q} for tid, q in rows]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
