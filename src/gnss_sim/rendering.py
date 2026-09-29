"""Render only observed N/E/U input; never load labels."""
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

AXES = ("N", "E", "U")
TICKS = (0, 60, 120, 180, 240, 300, 364)


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
