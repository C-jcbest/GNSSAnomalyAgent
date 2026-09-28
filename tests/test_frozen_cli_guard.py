import sys

import pytest

from gnss_sim.cli import main


@pytest.mark.parametrize("filename", ["predictions.jsonl", "report.json", "run.json"])
def test_numerical_cli_refuses_to_overwrite_even_partial_run(tmp_path, monkeypatch, filename):
    directory = tmp_path / "sr"
    directory.mkdir()
    artifact = directory / filename
    artifact.write_bytes(b"preserve these bytes")
    monkeypatch.setattr(sys, "argv", ["gnss-sim", "numerical", "--method", "sr",
                                    "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert artifact.read_bytes() == b"preserve these bytes"
    assert {item.name for item in directory.iterdir()} == {filename}
