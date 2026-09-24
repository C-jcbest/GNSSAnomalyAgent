from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from gnss_anomaly.contracts import Window
from gnss_anomaly.storage import read_json


@pytest.fixture
def config():
    return read_json(Path(__file__).parents[1] / "configs/experiment.json")


@pytest.fixture
def window():
    def make(values=None, length=48):
        values = np.zeros((length, 3)) if values is None else np.asarray(values)
        return Window(
            case_id="opaque",
            timestamps=[
                datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
                for i in range(len(values))
            ],
            values=[[float(x) if np.isfinite(x) else None for x in row] for row in values],
        )

    return make
