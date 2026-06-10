"""Central configuration for conversation-analyser.

Thresholds, model names, and score weights live here so callers tune behaviour
in one place. The taxonomy band thresholds and embeddings constants are ported
verbatim from the ISYS6020 marking_pipeline (then forked — see the design spec).
"""

from __future__ import annotations

import os

# --- LLM models ---
CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"

# --- LLM-segment chunking (long unstructured transcripts) ---
# The segmenter labels human turns in one Haiku call, but that call only sees the
# first SEGMENT_CHUNK_CHARS characters — so a long unstructured transcript would
# get only its opening labelled. Longer transcripts are split on paragraph
# boundaries into chunks of this size, classified separately, then concatenated so
# the WHOLE transcript is segmented before aggregation. Cleanly-labelled
# transcripts take the heuristic path and never hit this.
SEGMENT_CHUNK_CHARS = int(
    os.getenv("CONVERSATION_ANALYSER_SEGMENT_CHUNK_CHARS", "36000")
)
# Guard on the number of chunks (= Haiku calls) per transcript, so a pathological
# input can't silently fan out into many calls. 0 = unlimited. Raise per run, e.g.
# `CONVERSATION_ANALYSER_SEGMENT_MAX_CHUNKS=0 conversation-analyser …`; when the cap
# bites it analyses up to the cap and emits a note (never silently drops the tail).
SEGMENT_MAX_CHUNKS = int(os.getenv("CONVERSATION_ANALYSER_SEGMENT_MAX_CHUNKS", "12"))

# --- Engagement-spectrum heuristic band mapping (ported) ---
# Consumed by taxonomy.suggested_band. Order of evaluation matters; see taxonomy.py.
ENGAGEMENT_BAND_THRESHOLDS = {
    "Critical": {
        "critical_thinking_min": 0.40,
        "longest_engaged_chain_min": 3,
        "pushback_regex_hits_min": 2,
    },
    "Iterative": {
        "critical_thinking_min": 0.25,
        "critical_thinking_max": 0.40,
        "fu_plus_ex_min": 0.40,
    },
    "Directed": {
        "critical_thinking_min": 0.10,
        "critical_thinking_max": 0.25,
    },
    "Delegator": {
        "delegation_min": 0.40,
        "critical_thinking_max": 0.10,
    },
    "One-Shot": {
        "max_user_turns": 3,
    },
}

# Ordinal rank of each band, low → high engagement. Used to assert score↔band
# agreement and to bin a continuous score back to a band for reference.
BAND_ORDER = ("One-Shot", "Delegator", "Directed", "Iterative", "Critical")

FILLER_HEAVY_RATIO = 0.30
HIGH_REPEAT_SIMILARITY = 0.75

# --- Embeddings ---
EMBED_MODEL = "all-MiniLM-L6-v2"

# --- Sessionization ---
# Split one conversation into sub-sessions on idle gaps >= this many minutes,
# but only when turn timestamps are present.
IDLE_GAP_MIN = 30.0

# --- Critical-thinking composite score (0-100) ---
# score = 100 * clamp( w_ratio*ct_ratio + w_chain*norm(chain) + w_pushback*norm(pushback)
#                      - w_deleg*delegation_ratio - w_filler*filler_ratio , 0, 1)
# Calibrated 2026-05-23 against 45 real labelled transcripts to make the score
# monotonic with the authoritative band: band-order inversions dropped 5.1% -> 2.4%
# (medians Delegator 10 / Directed 27 / Iterative 39 / Critical 57). Re-tune with
# scripts/calibrate_ct_score.py --grid --signals <dir>.
CT_SCORE_WEIGHTS = {
    "ratio": 0.70,
    "chain": 0.10,
    "pushback": 0.10,
    "deleg": 0.20,
    "filler": 0.20,
}
CHAIN_CAP = 5  # longest_engaged_chain saturates here when normalised
PUSHBACK_CAP = 4  # pushback hits saturate here when normalised

# --- Serve ---
DEFAULT_PORT = 8009
