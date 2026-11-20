# gta6-trophy-rarity-forecast-01-entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/StarsN/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/StarsN/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for Grand Theft Auto VI's PlayStation 5 trophy earn rates on Exophase. Given a case — the game's 55 trophies and every read the house recorded of the game's Exophase page before a checkpoint — it writes, for each trophy, five quantiles (5%, 25%, 50%, 75%, 95%) of the earn rate at the case's target: the percent of Exophase-tracked players who will hold that trophy a week after launch. The score is the mean pinball loss on the logit scale, so the model is built to be calibrated, not just close: an honest spread around a well-placed median.

## How it works

The history is a step function — Exophase refreshes its counts on its own clock, so the recorder's reads repeat one page state many times. The program compresses the reads to the distinct page states and sets aside the pre-release states (a few hundred reviewers whose rates sit on a different trajectory from the launch population's). Each trophy is then classified from its name, description and tier — story chapter, story finale, named mission, stranger strand, grind, incidental, the 100%-completion gold, the platinum, the saturated opening mission — and the class carries a prior for how fast the rate's logit grows per unit of log-time, a ceiling, and a spread component.

Each trophy's median is its last observed rate grown on the logit scale by a mix of three estimates: its own weighted least-squares slope against `u = ln(1 + hours since release / 12)` over the distinct post-release states (weighted by the square root of players tracked, shrunk toward the class prior, winsorised so one lumpy Exophase refresh cannot dominate); the pooled growth of the trophies sitting at a similar rate level, measured over the later part of the window and clamped so the pool moderates rather than amplifies; and the class prior slope. Total growth is capped and floored, so no trophy's odds multiply without limit and no checkpoint dip becomes a crash.

The story chain gets a structural pass: each chapter's rate at the target is also forecast as a pass-through fraction of the previous chapter's — the fraction rising down the chain, because the players who get far are the committed ones — blended with a catch-up dynamic measured from the current earner ratio when that ratio is measurable, and the chain's medians are made non-increasing. Story missions and stranger strands with barely any earners yet get a floor from the finale's forecast (nobody finishes the story without passing its missions). Trophies with no earners at all in the tracked population — the platinum, the 100% trophy, the longest grinds on day two — have no slope to fit: their median comes from a class prior path scaled by the days since release at the target, with a wide spread. The platinum is capped at the 100%-completion forecast, since it is earned only by players who hold that trophy too. If a case has no usable post-release states at all (the practice shape: only the stale pre-release page), a dilution fallback applies — the tracked population multiplies over the horizon while earners crawl, so rates fall, with a wide spread.

Spreads combine a base term, the fit's residual scale grown by the extrapolation distance, and a class term, floored so no trophy is forecast with false confidence. Quantiles are `sigmoid(median_logit + z·sigma)`, rounded to 4 decimals.

Design notes and calibration evidence are in the source's docstring. The calibration was sanity-checked against the live Exophase page two days after launch (fetched 2026-11-20) and against published roadmap reporting for the game (55 offline trophies, roughly 200 hours to the platinum, a lenient 100% checklist): the model's day-7 medians put Chapter 1 near a third of tracked players, the finale in single digits, and the platinum well under a tenth of a percent.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. The program reads every `<id>.in` case in the input directory (schema `makefun/gta6-trophies-case/v1`) and writes one `<id>.out` per case to the output directory, each `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`. With no environment set it reads `/input` and writes `/output`, as the grader runs it. Two runs on the same case write the same bytes; a case takes well under a second.

---

<sub>Entered on [make.fun](https://make.fun/?ref=vvry49c9&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
