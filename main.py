"""GTA VI launch-day Twitch concurrent-viewer forecaster.

The first hour of a launch says a lot about how much of the world has
unlocked the game yet. A rolling-midnight console launch opens small --
only the earliest time zones are in -- and then ramps as regions unlock
(US East around hour 6, US West around hour 9) before reaching its
evening peak. A simultaneous global launch, by contrast, opens near its
peak right away. So instead of assuming one fixed daily shape, this
forecaster learns the shape from past launches, conditioned on how big
the opening hour was and how crowded it was (viewers per channel).

For each later hour it combines two views of those past launches: the
handful of nearest neighbours in opening size and crowdedness, and a
straight-line trend of the hour's growth against opening size across
all usable launches. Launches whose unlock pattern differs from this
launch's pattern are pushed further away in the neighbour match, so a
rolling launch is mostly compared with other rolling launches. The two
views are blended into a centre and a spread, and each requested
quantile is the opening-hour level times exp(centre + z * spread).
When no past launch covers an hour, a widening fallback spread is used
instead. Standard library only, no network access, fully deterministic.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

NEIGHBOURS = 8
NEIGHBOUR_WEIGHT = 0.75
SPREAD_SCALE = 0.75
SIGMA_FLOOR = 0.05
FALLBACK_SIGMA = 0.15
TYPE_PENALTY = 2.0
DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

_ROLLING_MARKS = (
    "rolling midnight",
    "midnight local",
    "12 am local",
    "12am local",
    "local midnight",
    "midnight in each",
    "(rolling",
    "rolling;",
)


def parse_time(text):
    """Parse an ISO 8601 timestamp; None when unreadable.

    A trailing Z is accepted, and naive datetimes are treated as UTC.
    """
    if not isinstance(text, str):
        return None
    value = text.strip()
    if not value:
        return None
    if value.endswith(("Z", "z")):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def normal_quantile(p):
    """Inverse standard normal CDF (Abramowitz & Stegun 26.2.23).

    Accurate to about 1e-4. Raises ValueError outside (0, 1), and
    returns exactly 0.0 for p == 0.5. Uses symmetry below the median.
    """
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise ValueError("p must be a number strictly between 0 and 1")
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -normal_quantile(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    numerator = 2.515517 + t * (0.802853 + t * 0.010328)
    denominator = 1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308))
    return t - numerator / denominator


def round_half_up(value, places=2):
    """Round to `places` decimals, halves away from zero.

    Returned as a plain number: an int when the result is whole.
    """
    factor = 10.0 ** places
    scaled = math.floor(abs(value) * factor + 0.5) / factor
    if value < 0:
        scaled = -scaled
    if scaled == int(scaled):
        return int(scaled)
    return scaled


def median(values):
    """Ordinary median of a list of numbers."""
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def iqr_sigma(values):
    """Robust spread from the interquartile range of log ratios.

    Takes the values at the quarter and three-quarter positions and
    scales their gap by 1.349, floored at SIGMA_FLOOR.
    """
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return SIGMA_FLOOR
    lower = ordered[count // 4]
    upper = ordered[(3 * count) // 4]
    return max(SIGMA_FLOOR, (upper - lower) / 1.349)


def release_day_curve(comparable):
    """Map a past launch's hourly points onto release-day hours 1..24.

    Returns ({hour: viewers}, channels at hour 1 or None). A point is
    kept only when its viewers are positive, its timestamp parses, its
    rounded hour lies in 1..24, and it falls within 1800 seconds of
    that hour's start.
    """
    curve = {}
    channels_1 = None
    if not isinstance(comparable, dict):
        return curve, None
    day_start = parse_time(comparable.get("day_start"))
    if day_start is None:
        return curve, None
    hourly = comparable.get("hourly") or []
    for point in hourly:
        if not isinstance(point, dict):
            continue
        viewers = point.get("viewers")
        if isinstance(viewers, bool) or not isinstance(viewers, (int, float)):
            continue
        if viewers <= 0:
            continue
        stamp = point.get("from", point.get("at"))
        moment = parse_time(stamp)
        if moment is None:
            continue
        offset = (moment - day_start).total_seconds()
        hour = int(round(offset / 3600.0)) + 1
        if hour < 1 or hour > 24:
            continue
        if abs(offset - (hour - 1) * 3600.0) > 1800.0:
            continue
        curve[hour] = viewers
        if hour == 1:
            channels = point.get("channels")
            if (
                isinstance(channels, (int, float))
                and not isinstance(channels, bool)
                and channels > 0
            ):
                channels_1 = channels
            else:
                channels_1 = None
    return curve, channels_1


def first_hour_print(case):
    """Hour 1 (viewers, channels) for the launch being forecast.

    Prefers the case's "known" rows for hour 1 with positive viewers,
    else the last positive-viewers row of "history_hourly", else
    (1.0, None). Channels are None unless a positive number.
    """
    for row in case.get("known") or []:
        if not isinstance(row, dict):
            continue
        if row.get("hour") != 1:
            continue
        viewers = row.get("viewers")
        if isinstance(viewers, bool) or not isinstance(viewers, (int, float)):
            continue
        if viewers <= 0:
            continue
        channels = row.get("channels")
        if (
            isinstance(channels, (int, float))
            and not isinstance(channels, bool)
            and channels > 0
        ):
            return viewers, channels
        return viewers, None
    for row in reversed(case.get("history_hourly") or []):
        if not isinstance(row, dict):
            continue
        viewers = row.get("viewers")
        if isinstance(viewers, bool) or not isinstance(viewers, (int, float)):
            continue
        if viewers <= 0:
            continue
        channels = row.get("channels")
        if (
            isinstance(channels, (int, float))
            and not isinstance(channels, bool)
            and channels > 0
        ):
            return viewers, channels
        return viewers, None
    return 1.0, None


def crowdedness(viewers, channels):
    """log(viewers / channels), or None when either side is unusable."""
    if isinstance(viewers, bool) or not isinstance(viewers, (int, float)):
        return None
    if isinstance(channels, bool) or not isinstance(channels, (int, float)):
        return None
    if viewers <= 0 or channels <= 0:
        return None
    return math.log(viewers / channels)


def unlock_type(text):
    """Classify unlock wording as "rolling" or "simultaneous", else None.

    Rolling marks are checked first, case-insensitively.
    """
    if not isinstance(text, str):
        return None
    lowered = text.lower()
    for mark in _ROLLING_MARKS:
        if mark in lowered:
            return "rolling"
    if "simultaneous" in lowered or "single global" in lowered:
        return "simultaneous"
    return None


def forecast_case(case, comparables=None):
    """Forecast hours 2..24 of one case as five quantiles per hour."""
    levels = case.get("levels")
    if not (
        isinstance(levels, list)
        and levels
        and all(
            isinstance(level, (int, float))
            and not isinstance(level, bool)
            and 0.0 < level < 1.0
            for level in levels
        )
    ):
        levels = list(DEFAULT_LEVELS)
    p1_raw, channels_1 = first_hour_print(case)
    p1 = max(1.0, float(p1_raw))
    target_type = unlock_type(case.get("notes"))

    if comparables is None:
        comparables = case.get("comparables") or []
    pool = {}
    pool_channels = {}
    pool_types = {}
    for comparable in comparables:
        if not isinstance(comparable, dict):
            continue
        name = comparable.get("name")
        if not isinstance(name, str):
            continue
        curve, hour1_channels = release_day_curve(comparable)
        if 1 not in curve:
            continue
        pool[name] = curve
        pool_channels[name] = hour1_channels
        pool_types[name] = unlock_type(comparable.get("unlock"))

    hours = []
    for row in case.get("forecast") or []:
        if not isinstance(row, dict):
            continue
        hour = row.get("hour")
        if isinstance(hour, bool) or not isinstance(hour, int):
            continue
        if 2 <= hour <= 24:
            hours.append(hour)
    hours.sort()

    rows = []
    for hour in hours:
        quantiles = forecast_hour(
            p1, channels_1, hour, pool, pool_channels, pool_types,
            target_type, levels,
        )
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


# Short aliases for the same per-case forecaster.
forecast = forecast_case
predict_case = forecast_case
predict = forecast_case


def forecast_hour(p1, channels_1, hour, pool, pool_channels, pool_types=None,
                  target_type=None, levels=None):
    """Quantiles for one hour of the launch day, learned from past launches.

    pool maps launch name -> {hour: viewers}; pool_channels maps name ->
    hour-1 channels; pool_types maps name -> unlock type ("rolling",
    "simultaneous" or None). The target's own unlock type penalises
    neighbours whose unlock pattern differs from it. Returns the five
    quantiles for `levels` (default 5/25/50/75/95%), each clipped to
    [0.01, 1e9], rounded half-up to 2 decimals and non-decreasing.
    """
    if levels is None:
        levels = list(DEFAULT_LEVELS)
    z_scores = [normal_quantile(level) for level in levels]
    p1 = max(1.0, float(p1))
    log_p1 = math.log(p1)
    busy_1 = crowdedness(p1, channels_1)

    usable = []
    for name, curve in (pool or {}).items():
        if not isinstance(curve, dict):
            continue
        if 1 not in curve or hour not in curve:
            continue
        usable.append((name, curve))

    if not usable:
        spread = FALLBACK_SIGMA * math.sqrt(hour - 1)
        raw = [p1 * math.exp(z * spread) for z in z_scores]
    else:
        # Neighbour view: nearest launches in opening size and
        # crowdedness, with an unlock-type mismatch penalty.
        scored = []
        for (name, curve) in usable:
            first = curve[1]
            size_gap = abs(math.log(first) - log_p1)
            busy_pool = crowdedness(first, (pool_channels or {}).get(name))
            if busy_1 is None or busy_pool is None:
                distance = size_gap
            else:
                busy_gap = abs(busy_pool - busy_1)
                distance = math.sqrt(size_gap * size_gap + busy_gap * busy_gap)
            if target_type is not None and (pool_types or {}).get(name) != target_type:
                distance += TYPE_PENALTY
            scored.append((distance, name, curve))
        scored.sort(key=lambda item: (item[0], item[1]))
        nearest = scored[:NEIGHBOURS]
        ratios = [math.log(curve[hour] / curve[1]) for (_, _, curve) in nearest]
        near_median = median(ratios)
        near_spread = iqr_sigma(ratios)

        # Trend view: least-squares line of log ratio on log opening
        # size across every usable launch.
        xs = [math.log(curve[1]) for (_, curve) in usable]
        ys = [math.log(curve[hour] / curve[1]) for (_, curve) in usable]
        count = len(xs)
        mean_x = sum(xs) / count
        mean_y = sum(ys) / count
        sxx = sum((x - mean_x) ** 2 for x in xs)
        sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        slope = sxy / sxx if sxx > 0 else 0.0
        intercept = mean_y - slope * mean_x
        trend_median = intercept + slope * log_p1
        residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        trend_spread = max(
            SIGMA_FLOOR,
            math.sqrt(sum(r * r for r in residuals) / max(1, count - 2)),
        )

        centre = NEIGHBOUR_WEIGHT * near_median + (1.0 - NEIGHBOUR_WEIGHT) * trend_median
        spread = (
            NEIGHBOUR_WEIGHT * near_spread
            + (1.0 - NEIGHBOUR_WEIGHT) * trend_spread
        ) * SPREAD_SCALE
        raw = [p1 * math.exp(centre + z * spread) for z in z_scores]

    quantiles = []
    previous = None
    for value in raw:
        clipped = min(1e9, max(0.01, value))
        rounded = round_half_up(clipped, 2)
        if previous is not None and rounded < previous:
            rounded = previous
        quantiles.append(rounded)
        previous = rounded
    return quantiles


def main():
    """Read every *.in case, forecast it, write one *.out per case."""
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR") or "/input")
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(input_dir.glob("*.in"), key=lambda path: path.name):
        with open(in_path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        out_path = output_dir / (in_path.stem + ".out")
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
