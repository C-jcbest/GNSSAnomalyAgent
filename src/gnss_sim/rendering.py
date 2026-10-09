"""Render only observed N/E/U input; never load labels."""
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

AXES = ("N", "E", "U")
TICKS = (0, 60, 120, 180, 240, 300, 364)
POINT_CONTEXT = {"half_window_days": 14, "panels_per_page": 4,
                 "width_pixels": 1800, "panel_height_pixels": 300,
                 "values": "raw displacement_mm; every day; independent panel y limits"}


def render_case(case, path):
    """Draw three aligned panels with global day indices and millimetre axes."""
    values = np.asarray(case.displacement_mm, dtype=float)
    if values.shape != (365, 3) or not np.isfinite(values).all():
        raise ValueError("expected finite 365-day input")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.linewidth": 0.8, "savefig.facecolor": "white"}):
        fig, panels = plt.subplots(3, 1, figsize=(12, 8), dpi=150, sharex=True,
                                   facecolor="white")
        try:
            for i, (axis, name, panel) in enumerate(zip(AXES, ("North", "East", "Up"), panels)):
                panel.plot(np.arange(365), values[:, i], color="#233b58", linewidth=1)
                panel.set_xlim(0, 364)
                panel.set_xticks(TICKS)
                panel.tick_params(axis="x", labelbottom=True)
                panel.set_ylabel(f"{axis} displacement (mm)")
                panel.set_title(f"{axis} - {name}", loc="left", fontweight="bold")
                panel.margins(y=0.08)
            panels[-1].set_xlabel("Global day index (0-based)")
            fig.subplots_adjust(left=0.12, right=0.985, bottom=0.075, top=0.965, hspace=0.42)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            fig.savefig(path, format="png", dpi=150, facecolor="white")
        finally:
            plt.close(fig)


def render_point_context(case, items, directory):
    """Observation-defined views for every candidate; no labels or injection metadata."""
    values = np.asarray(case.displacement_mm, dtype=float)
    if values.shape != (365, 3) or not np.isfinite(values).all():
        raise ValueError("expected finite 365-day input")
    if len({c["id"] for c in items}) != len(items) or any(
        c["axis"] not in AXES or type(c["start"]) is not int
        or c["start"] != c["end"] or not 0 <= c["start"] <= 364 for c in items
    ):
        raise ValueError("Unique point candidates with valid observation indices required")
    paths = []
    for offset in range(0, len(items), POINT_CONTEXT["panels_per_page"]):
        batch = items[offset:offset + POINT_CONTEXT["panels_per_page"]]
        path = directory / f"page-{offset // POINT_CONTEXT['panels_per_page'] + 1:03d}.png"
        if path.exists():
            raise FileExistsError(path)
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                             "axes.linewidth": 0.8, "savefig.facecolor": "white"}):
            fig, panels = plt.subplots(len(batch), 1, figsize=(12, 2 * len(batch)),
                                      dpi=150, squeeze=False, facecolor="white")
            try:
                for c, panel in zip(batch, panels[:, 0]):
                    day = c["start"]
                    margin = POINT_CONTEXT["half_window_days"]
                    lo, hi = max(0, day-margin), min(364, day+margin)
                    x = values[lo:hi+1, AXES.index(c["axis"])]
                    panel.plot(np.arange(lo, hi+1), x, "o-", color="#233b58",
                               linewidth=1, markersize=3)
                    panel.axvline(day, color="#bb7500", linestyle="--", linewidth=1)
                    panel.scatter([day], [values[day, AXES.index(c["axis"])]],
                                  facecolors="none", edgecolors="#bb7500", s=90, zorder=3)
                    panel.set_xlim(lo-.5, hi+.5)
                    panel.set_xticks(sorted({lo, hi, day, *range(lo, hi+1, 4)}))
                    panel.set_ylabel(f'{c["axis"]} (mm)')
                    panel.set_xlabel("Global day index (0-based); raw observations")
                    panel.set_title(f'{c["id"]} / {c["axis"]} / candidate day {day} '
                                    "(proposal, not a label)", loc="left")
                    panel.margins(y=.12)
                fig.tight_layout()
                directory.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, format="png", dpi=150)
            finally:
                plt.close(fig)
        paths.append(path)
    return paths
