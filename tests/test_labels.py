from neurovision.data.labels import ANEURYSM_PRESENT, LOCATIONS, TARGETS


def test_targets_are_14_unique_names_with_aneurysm_present_last():
    assert len(TARGETS) == 14
    assert len(set(TARGETS)) == 14
    assert TARGETS[-1] == "Aneurysm Present"


def test_locations_and_aneurysm_present_split_targets():
    assert TARGETS[:13] == LOCATIONS
    assert TARGETS[13] == ANEURYSM_PRESENT


def test_target_order_matches_kaggle_label_cols():
    assert TARGETS[:4] == (
        "Left Infraclinoid Internal Carotid Artery",
        "Right Infraclinoid Internal Carotid Artery",
        "Left Supraclinoid Internal Carotid Artery",
        "Right Supraclinoid Internal Carotid Artery",
    )
    assert TARGETS[11:] == ("Basilar Tip", "Other Posterior Circulation", "Aneurysm Present")
