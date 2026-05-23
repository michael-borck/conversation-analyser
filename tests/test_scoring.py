from conversation_analyser import scoring, taxonomy


def _score(labels, chain, pushback):
    r = taxonomy.compute_ratios(labels)
    band = taxonomy.suggested_band(
        ratios=r,
        longest_chain=chain,
        pushback_hits=pushback,
        user_turn_count=len(labels),
        labels=labels,
    )
    return scoring.critical_thinking_score(
        ratios=r, longest_chain=chain, pushback_hits=pushback, band=band
    )


def test_score_bounds_and_band_passthrough():
    ct = _score(["CH", "EX", "CH", "EX", "CH"], 5, 3)
    assert 0.0 <= ct.score <= 100.0
    assert ct.band == "Critical"


def test_critical_scores_higher_than_delegator():
    crit = _score(["CH", "EX", "CH", "EX", "CH"], 5, 3)
    deleg = _score(["DG", "DG", "DG", "DG"], 0, 0)
    assert crit.score > deleg.score
    assert deleg.score == 0.0  # pure delegation, penalties clamp to floor


def test_components_present():
    ct = _score(["CH", "EX"], 2, 1)
    assert set(ct.components) == {"ratio", "chain", "pushback", "deleg_penalty", "filler_penalty"}
