import numpy as np
from run_landslide_artifact_points import decode, scores


def test_unsupported_known_artifact_remains_a_false_negative_and_missing_is_excluded():
    bank = {
        "actual": np.array([1, -1, 0]),
        "activity": np.array([1, -1, 0]),
        "observed": np.array([True, False, True]),
        "supported": np.array([False, False, True]),
    }
    probability = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    prediction = decode(probability, bank["supported"], 0.5)
    result = scores({"case": bank}, {"case": prediction}, {"case": probability})
    assert prediction.tolist() == [-1, -1, 0]
    assert result["all"]["presence"]["fn"] == 1
    assert result["all"]["presence"]["known_days"] == 2
    assert result["all"]["presence"]["unknown_on_known"] == 1


def test_correct_presence_with_wrong_type_is_a_type_error_even_during_deformation():
    bank = {
        "actual": np.array([1]),
        "activity": np.array([1]),
        "observed": np.array([True]),
        "supported": np.array([True]),
    }
    probability = np.array([[0.1, 0.2, 0.7]])
    prediction = decode(probability, bank["supported"], 0.5)
    result = scores({"case": bank}, {"case": prediction}, {"case": probability})
    assert result["deformation_days"]["presence"]["tp"] == 1
    assert result["all"]["instrument_offset"]["fn"] == 1
    assert result["all"]["spike"]["fp"] == 1
