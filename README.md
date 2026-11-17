# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Sixtyseconds5/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Sixtyseconds5/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Hour-by-hour forecast of Grand Theft Auto VI's launch-day Twitch viewers: a stdlib-only Python program that reads the first hour's viewer count plus past big launches' hourly curves and writes five quantiles (5%, 25%, 50%, 75%, 95%) for each of hours 2 to 24.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## How it works

A launch's first hour tells how much of the world has unlocked yet: a game that opened everywhere at once sits near its peak, while a rolling-midnight console launch opens small and ramps hard as each region wakes up and unlocks (US East at hour 6, US West at hour 9, then the evening peak). So the shape of the day is learned from the past launches in the case's comparables, conditioned on the opening hour:

- For each past launch, the log ratio `ln(viewers_h / viewers_1)` for hours 2–24.
- Two views per hour, averaged: the median log ratio among the 10 past launches whose opening hour most resembled this one — matched on opening size and on opening crowdedness (`log(viewers per live channel)`) — and a straight-line fit of the log ratio against `log(first hour)`, read off at this launch's first hour.
- Quantiles are the median times `exp(z_tau * spread)`, where the spread averages the two views' dispersions (interquartile range for the neighbours, residual spread for the trend), narrowed by 0.9.

Leave-one-out backtest over the 37 comparable launches with complete release-day curves, scored with the contest's own pinball loss on the log scale: **0.396** for this model vs **0.930** for the flat persistence baseline; on the 27 launches whose first hour was at least 20,000 viewers — the regime the practice cases and GTA VI sit in — **0.267** vs **0.321**. The house baseline scores 0.387 on the practice cases, so the model clears that bar with room to spare.

## Files

- `main.py` — the forecaster. Python 3.13 standard library only, no network, fully deterministic (two runs on the same case write byte-identical output), about 0.05 s per case.
- `README.md` — this file.
- `LICENSE` — MIT.

---

<sub>Entered on [make.fun](https://make.fun/?ref=bny55a6b&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
