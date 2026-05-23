from conversation_analyser import taxonomy


def test_ratios_chain_and_counts():
    labels = ["NQ", "CH", "EX", "FU", "DG", "AC", "MT"]
    r = taxonomy.compute_ratios(labels)
    assert r["critical_thinking"] == round(2 / 7, 2)
    assert r["delegation"] == round(1 / 7, 2)
    assert r["filler"] == round(2 / 7, 2)
    assert taxonomy.longest_engaged_chain(["CH", "EX", "FU", "NQ", "CH"]) == 3
    assert taxonomy.label_counts(labels)["CH"] == 1


def test_band_one_shot():
    labels = ["DG"]
    r = taxonomy.compute_ratios(labels)
    band = taxonomy.suggested_band(
        ratios=r, longest_chain=0, pushback_hits=0, user_turn_count=1, labels=labels
    )
    assert band == "One-Shot"


def test_band_delegator():
    labels = ["DG", "DG", "DG", "DG"]
    r = taxonomy.compute_ratios(labels)
    band = taxonomy.suggested_band(
        ratios=r, longest_chain=0, pushback_hits=0, user_turn_count=4, labels=labels
    )
    assert band == "Delegator"


def test_band_critical():
    labels = ["CH", "EX", "CH", "EX", "CH"]
    r = taxonomy.compute_ratios(labels)
    band = taxonomy.suggested_band(
        ratios=r, longest_chain=5, pushback_hits=3, user_turn_count=5, labels=labels
    )
    assert band == "Critical"


def test_filler_heavy():
    assert taxonomy.filler_heavy({"filler": 0.5}) is True
    assert taxonomy.filler_heavy({"filler": 0.1}) is False


def test_classify_turns_mocked(monkeypatch):
    def fake_call(*, system, user, model):
        return {
            "labels": [
                {"index": 0, "label": "CH", "rationale": "pushes back"},
                {"index": 1, "label": "DG", "rationale": "hand-off"},
            ]
        }

    monkeypatch.setattr(taxonomy, "_call_classifier", fake_call)
    out = taxonomy.classify_turns([{"prompt": "why?"}, {"prompt": "write it"}])
    assert [o["label"] for o in out] == ["CH", "DG"]
    assert out[0]["rationale"] == "pushes back"


def test_classify_turns_invalid_label_collapses(monkeypatch):
    monkeypatch.setattr(
        taxonomy, "_call_classifier", lambda **k: {"labels": [{"index": 0, "label": "ZZ"}]}
    )
    out = taxonomy.classify_turns([{"prompt": "x"}])
    assert out[0]["label"] == "NQ"
