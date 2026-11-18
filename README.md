# Forecast GTA VI's launch-day Twitch viewers hour by hour

[![make.fun bounty](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![make.fun rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Sixtyseconds5/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![make.fun reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Sixtyseconds5/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for the [make.fun](https://make.fun) bounty **gta6-twitch-ccu-forecast-01** — forecast GTA VI's launch-day Twitch concurrent viewers hour by hour ($250). Given the real first-hour viewer count, the program forecasts hours 2–24 as five quantiles (5%, 25%, 50%, 75%, 95%) per hour, borrowing the shape of the day from past big game launches.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each `*.in` case in the input directory produces one `<id>.out` file with 23 rows (hours 2–24). Python standard library only; no network access; deterministic — two runs on the same case write byte-identical output.

## How it works

The first hour tells how much of the world has unlocked the game yet, so the day's shape is learned from past launches rather than assumed. For each hour, two views are combined:

- **Nearest neighbours:** the 8 past launches closest in opening size and crowdedness (viewers per channel), measured in log space. Launches whose unlock pattern differs from this launch's — rolling midnight versus a simultaneous global unlock — are pushed 2.0 further away in the neighbour match, so a rolling launch is mostly compared with other rolling launches. The centre is the median of the neighbours' log growth ratios, and the spread is their interquartile range scaled to a standard deviation.
- **Log-size trend:** a least-squares line of log growth ratio against log opening size across every usable past launch, read off at this launch's opening size, with the spread taken from the residuals.

The two views are blended 75/25 into a centre and a spread, and each quantile is the first-hour level times exp(centre + z × spread). When no past launch covers an hour, a widening fallback spread is used instead.

## Backtest

Leave-one-out over the 37 comparable launches with complete release-day curves in the kit's practice cases, scored with the contest's own pinball loss on the log scale: this model scores **0.384** overall, **0.235** on launches whose first hour was at least 20,000 viewers, and **0.174** on launches whose first hour was at least 100,000, versus the house baseline's 0.930 overall and 0.321 on the 20k+ launches. The previous entry scored 0.205388 on the practice cases against the house baseline's 0.387.

## Files

- `main.py` — the forecaster (Python standard library only)
- `README.md` — this file
- `LICENSE` — MIT
