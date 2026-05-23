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


def test_score_monotonic_with_band():
    """Calibration property: median score must rise with the authoritative band.

    Uses a representative synthetic corpus (no real student data); see
    scripts/calibrate_ct_score.py for the data-driven weight calibration.
    """
    from statistics import median

    from conversation_analyser import taxonomy
    from conversation_analyser.config import BAND_ORDER

    corpus = [
        (["DG", "DG", "DG", "NQ"], 0),
        (["NQ", "FU", "NQ", "AC"], 1),
        (["NQ", "EX", "FU", "EX", "NQ"], 1),
        (["NQ", "EX", "FU", "EX", "CH", "FU"], 2),
        (["CH", "EX", "CH", "EX", "CH"], 4),
        (["CH", "EX", "FU", "CH", "EX", "CH"], 5),
    ]
    by_band: dict[str, list[float]] = {}
    for labels, pushback in corpus:
        ratios = taxonomy.compute_ratios(labels)
        chain = taxonomy.longest_engaged_chain(labels)
        band = taxonomy.suggested_band(
            ratios=ratios, longest_chain=chain, pushback_hits=pushback,
            user_turn_count=len(labels), labels=labels,
        )
        ct = scoring.critical_thinking_score(
            ratios=ratios, longest_chain=chain, pushback_hits=pushback, band=band
        )
        by_band.setdefault(band, []).append(ct.score)

    present = [b for b in BAND_ORDER if b in by_band]
    medians = [median(by_band[b]) for b in present]
    assert len(present) >= 3, f"corpus should span several bands, got {present}"
    assert medians == sorted(medians), dict(zip(present, medians))
