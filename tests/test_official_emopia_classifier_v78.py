from scripts.run_official_emopia_classifier_v78 import macro_f1, summarize, target_for


def test_target_mapping_matches_russell_quadrants():
    assert target_for("ar_va", "Q3") == "Q3"
    assert target_for("arousal", "Q1") == "HA"
    assert target_for("arousal", "Q2") == "HA"
    assert target_for("arousal", "Q3") == "LA"
    assert target_for("valence", "Q1") == "HV"
    assert target_for("valence", "Q4") == "HV"
    assert target_for("valence", "Q2") == "LV"


def test_macro_f1_and_summary_do_not_hide_missing_class():
    rows = [
        {"target": "Q1", "prediction": "Q1", "token_count": 10},
        {"target": "Q2", "prediction": "Q1", "token_count": 20},
    ]
    result = summarize(rows, ["Q1", "Q2"])
    assert result["accuracy"] == 0.5
    assert result["macro_f1"] == macro_f1(
        ["Q1", "Q2"], ["Q1", "Q1"], ["Q1", "Q2"]
    )
    assert result["confusion_matrix"]["Q2"]["Q1"] == 1
    assert result["token_count"]["median"] == 15.0

