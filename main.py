"""Hour-by-hour forecast of a big game launch's Twitch viewers.

Reads one case per run from the input directory (schema
makefun/twitch-ccu-case/v1) and writes, for each case, a forecast of hours
2..24 of the launch day as five quantiles (5%, 25%, 50%, 75%, 95%) of the
category's concurrent viewers.

How it forecasts
---------------
The launch's first hour tells us how much of the world has unlocked yet:
a game that unlocked everywhere at once opens near its peak, while a
rolling-midnight console launch opens small and ramps hard as each region
wakes up and unlocks (US East at hour 6, US West at hour 9, then the
evening peak). So the shape of the day is learned from past launches,
conditioned on the size of the first hour:

* For every past launch in the case's comparables, take the release-day
  hourly curve and form the log ratio ln(viewers_h / viewers_1) for each
  hour h = 2..24.
* Two views of that ratio are combined for each hour:
    - neighbours: the median log ratio among the 10 past launches whose
      opening hour most resembled this launch's, matched on two counts
      at once -- the size of the first hour (in log space) and how
      crowded it was, log(viewers per live channel). A launch that
      opened with few viewers spread over few channels is still
      ramping; one that opened with many viewers crowded into few
      channels, or spread over very many, behaves differently;
    - trend: a straight-line fit of the log ratio against log(first hour)
      across all past launches, read off at this launch's first hour.
  The forecast median is the average of the two views; the spread is the
  average of their dispersions (interquartile range for the neighbours,
  residual spread for the trend), narrowed a little (x0.9) because the
  backtest showed the raw dispersions run slightly wide.
* Quantiles are the median times exp(z_tau * spread), with z_tau the
  standard normal quantile at each requested level.

If a case carries no usable comparable curve at all, the forecast falls
back to a flat median at the first hour's count with a spread that widens
like a random walk in log space.

Only the standard library is used, nothing is read from the network, and
every iteration is over sorted data, so two runs on the same case write
byte-identical output.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Neighbour count for the local view, weight of that view in the blend,
# and the spread narrowing the leave-one-out backtest favoured.
NEIGHBOURS = 10
NEIGHBOUR_WEIGHT = 0.5
SPREAD_SCALE = 0.9
SIGMA_FLOOR = 0.05

# Fallback spread (log units per square-root hour) when no comparable
# curve exists at all.
FALLBACK_SIGMA = 0.15


def parse_time(text):
    """Parse an ISO 8601 timestamp; None when it cannot be read."""
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def normal_quantile(p):
    """Inverse of the standard normal CDF (Abramowitz & Stegun 26.2.23).

    Accurate to about 1e-4 in the quantile, far tighter than the forecast
    needs; p = 0.5 returns exactly 0.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -normal_quantile(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    num = 2.515517 + t * (0.802853 + t * 0.010328)
    den = 1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308))
    return t - num / den


def round_half_up(value, places=2):
    """Round to `places` decimals, halves away from zero, as a number."""
    scaled = value * (10.0 ** places)
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / (10.0 ** places)
    return int(rounded) if rounded == int(rounded) else rounded


def median(values):
    ordered = sorted(values)
    count = len(ordered)
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def iqr_sigma(values):
    """Robust spread of log ratios from their interquartile range."""
    ordered = sorted(values)
    count = len(ordered)
    lower = ordered[count // 4]
    upper = ordered[(3 * count) // 4]
    return max(SIGMA_FLOOR, (upper - lower) / 1.349)


def release_day_curve(comparable):
    """Release-day data for a comparable: ({hour: viewers}, hour-1 channels).

    Points whose timestamp falls outside the 24 hours starting at the
    comparable's day_start are ignored; a point must carry a positive
    viewer count to count. The hour-1 channel count is None when the
    source did not record one.
    """
    day_start = parse_time(comparable.get("day_start"))
    if day_start is None:
        return {}, None
    curve = {}
    channels_1 = None
    for point in comparable.get("hourly") or []:
        viewers = point.get("viewers")
        if not isinstance(viewers, (int, float)) or viewers <= 0:
            continue
        moment = parse_time(point.get("from"))
        if moment is None:
            continue
        offset = (moment - day_start).total_seconds()
        hour = int(round(offset / 3600.0)) + 1
        if 1 <= hour <= 24 and abs(offset - (hour - 1) * 3600.0) <= 1800.0:
            curve[hour] = float(viewers)
            if hour == 1:
                channels = point.get("channels")
                if isinstance(channels, (int, float)) and channels > 0:
                    channels_1 = float(channels)
    return curve, channels_1


def first_hour_print(case):
    """Hour 1's (viewers, channels), viewers floored at 1.

    Falls back to the last whole-hour mean before the checkpoint, then to
    (1, None), so a case with a missing hour-1 print still gets a valid
    forecast. Channels are None when the case did not record any.
    """
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                channels = row.get("channels")
                if not isinstance(channels, (int, float)) or channels <= 0:
                    channels = None
                return float(viewers), channels
    hourly = case.get("history_hourly") or []
    if hourly:
        viewers = hourly[-1].get("viewers")
        if isinstance(viewers, (int, float)) and viewers > 0:
            channels = hourly[-1].get("channels")
            if not isinstance(channels, (int, float)) or channels <= 0:
                channels = None
            return float(viewers), channels
    return 1.0, None


def crowdedness(viewers, channels):
    """Log viewers per live channel at the opening hour, or None."""
    if channels is None or channels <= 0 or viewers <= 0:
        return None
    return math.log(viewers / channels)


def forecast_hour(p1, channels_1, hour, curves, channels):
    """Five quantiles for one hour, from the comparables' log ratios.

    `curves` maps a launch name to its release-day {hour: viewers} curve
    and `channels` maps it to its hour-1 live-channel count (or None).
    Returns a list of five numbers in the case's level order.
    """
    log_p1 = math.log(max(1.0, p1))
    busy_1 = crowdedness(p1, channels_1)
    usable = {
        name: curve
        for name, curve in curves.items()
        if 1 in curve and hour in curve
    }
    if not usable:
        spread = FALLBACK_SIGMA * math.sqrt(hour - 1)
        return [p1 * math.exp(z * spread) for z in Z_SCORES]

    # Neighbour view: the launches whose opening hour most resembled
    # this one, on opening size and on opening crowdedness together.
    def distance(name):
        size_gap = abs(math.log(usable[name][1]) - log_p1)
        busy = crowdedness(usable[name][1], channels.get(name))
        if busy_1 is None or busy is None:
            return size_gap
        return math.sqrt(size_gap * size_gap + (busy - busy_1) ** 2)

    ranked = sorted(usable, key=lambda name: (distance(name), name))
    near = ranked[:NEIGHBOURS]
    near_ratios = [
        math.log(usable[name][hour] / usable[name][1]) for name in near
    ]
    near_median = median(near_ratios)
    near_spread = iqr_sigma(near_ratios)

    # Trend view: straight-line fit of log ratio on log first hour.
    xs = [math.log(curve[1]) for curve in usable.values()]
    ys = [math.log(curve[hour] / curve[1]) for curve in usable.values()]
    count = len(xs)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    sxx = sum((x - mean_x) ** 2 for x in xs)
    slope = 0.0
    if sxx > 0.0:
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    trend_median = intercept + slope * log_p1
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    trend_spread = max(
        SIGMA_FLOOR,
        math.sqrt(sum(r * r for r in residuals) / max(1, count - 2)),
    )

    centre = NEIGHBOUR_WEIGHT * near_median + (1.0 - NEIGHBOUR_WEIGHT) * trend_median
    spread = (
        NEIGHBOUR_WEIGHT * near_spread + (1.0 - NEIGHBOUR_WEIGHT) * trend_spread
    ) * SPREAD_SCALE
    return [p1 * math.exp(centre + z * spread) for z in Z_SCORES]


# Standard normal quantiles at the levels this run scores; set per case
# in forecast_case before any hour is forecast.
Z_SCORES = [normal_quantile(level) for level in DEFAULT_LEVELS]


def forecast_case(case):
    global Z_SCORES
    levels = case.get("levels") or DEFAULT_LEVELS
    Z_SCORES = [normal_quantile(level) for level in levels]
    p1, channels_1 = first_hour_print(case)

    curves = {}
    channels = {}
    for comparable in case.get("comparables") or []:
        name = comparable.get("name")
        if not isinstance(name, str):
            continue
        curve, hour_1_channels = release_day_curve(comparable)
        if 1 in curve:
            curves[name] = curve
            channels[name] = hour_1_channels

    hours = []
    for row in sorted(case.get("forecast") or [], key=lambda r: r.get("hour", 0)):
        hour = row.get("hour")
        if not isinstance(hour, int) or hour < 2 or hour > 24:
            continue
        raw = forecast_hour(p1, channels_1, hour, curves, channels)
        quantiles = []
        previous = 0.0
        for value in raw:
            clipped = min(1e9, max(0.01, value))
            rounded = round_half_up(clipped)
            if rounded < previous:
                rounded = previous
            quantiles.append(rounded)
            previous = rounded
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
