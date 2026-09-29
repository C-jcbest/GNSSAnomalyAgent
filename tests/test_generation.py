import numpy as np


def test_native_sources_pairing_labels_and_observed_effects():
    from gnss_sim.generator import SIGMA_MM, generate_cases
    from gnss_sim.schemas import GenerationRequest

    request = GenerationRequest(seed=20260929, count=24, case_type="all")
    rows = list(generate_cases(request))  # Worker also checks hashes and TODS equivalence.
    assert rows == list(generate_cases(request))
    assert len(rows) == 24
    directions = set()
    for group in range(6):
        normal, normal_truth, _ = rows[group * 4]
        noise = np.asarray(normal.displacement_mm)
        # GutenTAG draws channels sequentially. No empirical normalization or clipping.
        rng = np.random.default_rng(normal_truth.noise_seed)
        expected = np.column_stack([rng.normal(0, 1, 365) for _ in range(3)]) * SIGMA_MM
        np.testing.assert_array_equal(noise, expected)
        for case, truth, plan in rows[group * 4:group * 4 + 4]:
            assert truth.measurement_noise_mm == normal.displacement_mm
            assert set(case.model_dump()) == {
                "schema_version", "case_id", "dates", "reference_coordinate_mm",
                "observed_coordinate_mm", "displacement_mm", "horizontal_offset_mm",
                "spatial_offset_mm",
            }
            observed = np.asarray(case.displacement_mm)
            delta = np.asarray(truth.anomaly_delta_mm)
            labels = np.asarray(truth.axis_labels)
            np.testing.assert_allclose(observed, noise + delta, atol=1e-14)
            np.testing.assert_array_equal(labels.max(axis=1), truth.native_labels)
            if not truth.events:
                assert not labels.any() and not delta.any()
                continue
            event = truth.events[0]
            axis, sign = plan["axis"], plan["sign"]
            directions.add((axis, sign))
            start, end = event.start_index, event.end_index
            assert 60 <= start <= end <= 304
            expected_labels = np.zeros((365, 3), dtype=int)
            expected_labels[start:end + 1, axis] = 1
            np.testing.assert_array_equal(labels, expected_labels)
            np.testing.assert_array_equal(np.delete(observed, axis, axis=1),
                                          np.delete(noise, axis, axis=1))
            np.testing.assert_array_equal(observed[:start], noise[:start])
            if event.type == "global_extremum":
                others = np.delete(observed[:, axis], start)
                assert sign * observed[start, axis] > (sign * others).max()
                assert event.task == "point" and start == end
                np.testing.assert_array_equal(observed[end + 1:], noise[end + 1:])
            elif event.type == "mean_shift":
                assert end - start + 1 == 14 and event.task == "range"
                np.testing.assert_allclose(delta[start:end + 1, axis], sign * 3 * SIGMA_MM[axis])
                np.testing.assert_array_equal(observed[end + 1:], noise[end + 1:])
            else:
                assert end - start + 1 == 90 and event.task == "range"
                assert delta[start, axis] == 0 and labels[start, axis] == 1
                assert not labels[end + 1:].any()  # Residual is not activity.
                np.testing.assert_allclose(delta[end:, axis], sign * 3 * SIGMA_MM[axis])
    assert directions == {(axis, sign) for axis in range(3) for sign in (-1, 1)}
