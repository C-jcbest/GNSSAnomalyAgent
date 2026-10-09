import pytest

from gnss_sim.landslide_stage_grounding_experiment import citation_audit, image_inventory


def inventory_fixture():
    views = []
    for index, bounds in enumerate(([0, 1094], [0, 409], [320, 774], [685, 1094])):
        views.append({"view": bounds, "image": f"raw-{index}.png",
                      "audit": {"panels": [{"xlim": bounds} for _ in range(3)]}})
    record = {"case_id": "case_9999", "views": views,
              "diagnostics": {str(w): {"image": f"aux-{w}.png"} for w in (31, 61, 91)}}
    images = [view["image"] for view in views] + [f"aux-{w}.png" for w in (31, 61, 91)]
    return record, images


def test_inventory_distinguishes_full_record_from_last_local_view():
    record, images = inventory_fixture()
    inventory = image_inventory(record, images, 1095)
    assert inventory["images"][0]["axis_days_inclusive"] == [0, 1094]
    assert inventory["images"][3]["axis_days_inclusive"] == [685, 1094]
    assert [image["id"] for image in inventory["images"]] == ["R1", "R2", "R3", "R4", "M31", "M61", "M91"]


def test_inventory_rejects_reordered_actual_images():
    record, images = inventory_fixture()
    images[0], images[3] = images[3], images[0]
    with pytest.raises(ValueError, match="actual seven-image order"):
        image_inventory(record, images, 1095)


def test_inventory_rejects_range_not_matching_rendered_axes():
    record, images = inventory_fixture()
    record["views"][2]["audit"]["panels"][0] = {"xlim": [685, 1094]}
    with pytest.raises(ValueError, match="actual frozen axes"):
        image_inventory(record, images, 1095)


def test_locators_do_not_certify_stage_or_allow_wrong_image_range():
    record, images = inventory_fixture()
    inventory = image_inventory(record, images, 1095)
    answer = {"stages": [
        {"start": 525, "stop": 610, "feature": "deceleration", "evidence": "[R4] [M61]"},
        {"start": 525, "stop": 610, "feature": "deceleration", "evidence": "[R3] [M61]"},
        {"start": 525, "stop": 610, "feature": "deceleration", "evidence": "[R1] [M61] [R9]"},
    ]}
    result = citation_audit(answer, inventory)
    assert [stage["locator_and_coverage_pass"] for stage in result["stages"]] == [False, True, False]
    assert result["unknown_image_id_stages"] == 1
    assert result["no_covering_raw_citation_stages"] == 1


def test_last_inclusive_axis_day_can_cover_exclusive_stage_stop():
    record, images = inventory_fixture()
    inventory = image_inventory(record, images, 1095)
    answer = {"stages": [{"start": 1055, "stop": 1095, "feature": "acceleration", "evidence": "[R4] [M31]"}]}
    assert citation_audit(answer, inventory)["locator_and_coverage_pass"] == 1
