import io
import threading
from functools import wraps
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .contracts import CHANNELS, Prediction, Window

PLOT_VERSION = "neu-grid-v2"
PLOT_LOCK = threading.RLock()


def serial_plot(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with PLOT_LOCK:
            return fn(*args, **kwargs)

    return wrapped


@serial_plot
def render(
    window: Window, candidates: Prediction | None = None, bounds: tuple[int, int] | None = None
) -> bytes:
    values = window.array()
    first, last = bounds or (0, len(values) - 1)
    if not 0 <= first <= last < len(values):
        raise ValueError("invalid crop")
    figure, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True, layout="constrained")
    x = np.arange(first, last + 1)
    for c, (axis, name) in enumerate(zip(axes, CHANNELS)):
        axis.plot(
            x, values[first : last + 1, c], color="#245980", linewidth=0.9, marker=".", markersize=2
        )
        axis.set_ylabel(f"{name} (mm)")
        axis.grid(alpha=0.2)
        if candidates:
            for event in candidates.events:
                if name in event.channels and event.start <= last and event.end >= first:
                    axis.axvspan(
                        max(first, event.start) - 0.3,
                        min(last, event.end) + 0.3,
                        alpha=0.2,
                        color="#dd9f33",
                    )
    ticks = np.unique(np.linspace(first, last, min(7, len(x))).astype(int))
    axes[-1].set_xticks(
        ticks, [f"{i}\n{window.timestamps[i].strftime('%m-%d %H:%M')}" for i in ticks]
    )
    if window.sampling_hours == 24:
        axes[-1].set_xticks(
            ticks,
            [
                f"{i}\n{window.timestamps[i].astimezone(ZoneInfo('Asia/Shanghai')):%Y-%m-%d}"
                for i in ticks
            ],
        )
    axes[-1].set_xlabel(
        "Day index / Beijing date (15:00); gaps are unobserved"
        if window.sampling_hours == 24
        else "Global hour index / UTC timestamp; gaps are unobserved"
    )
    figure.suptitle(
        "GNSS displacement" + ("; shaded = detector candidates, not truth" if candidates else "")
    )
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=120)
    plt.close(figure)
    return stream.getvalue()


def review_bounds(window: Window, candidates: Prediction) -> tuple[int, int] | None:
    if not candidates.events:
        return None
    return (
        max(0, min(e.start for e in candidates.events) - 12),
        min(len(window.values) - 1, max(e.end for e in candidates.events) + 12),
    )
