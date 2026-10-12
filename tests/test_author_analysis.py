import importlib.util

import numpy as np
import pytest

from gnss_research.upstream import ROOT, parse_strict_list


def test_author_parser_retains_prose_and_original_index_behavior():
    if not (ROOT / "vendor/upstream/visualtime/.git").exists():
        pytest.skip("run gnss-research setup-sources for the locked author integration")
    spec = importlib.util.spec_from_file_location(
        "author_analysis", ROOT / "scripts/replay_author_parser.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    functions, _ = module.load_author_functions()
    text = "Visual inspection detected a spike.\n[2, 99]"
    with pytest.raises(ValueError):
        parse_strict_list(text, "point", 10)
    points = functions["parse_output"](text, "global")
    mask = functions["point_to_vector"](points, len_vector=10)
    np.testing.assert_array_equal(np.flatnonzero(mask), [2])
    intervals = functions["parse_output"]("Ranges: [[2, 5], [8, 99]]", "trend")
    mask = functions["interval_to_vector"](intervals, start=0, end=10, pred=True)
    np.testing.assert_array_equal(np.flatnonzero(mask), [2, 3, 4, 5, 8, 9])
