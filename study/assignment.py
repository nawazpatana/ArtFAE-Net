"""Balanced stimulus assignment.

The 60 stimuli are split once into 3 sets of 20, interleaving the strata (stratum-sorted list, item i -> set
i mod 3), so each set covers every stratum about equally. Participant k (0-based arrival index) gets set k mod 3
(rotation), so after every 3 participants each stimulus has been seen exactly once. Within the set the order is
shuffled with a per-participant seed, R1/R3 left-right placement in the A/B step is randomised per trial, and the
2 attention-check trials are inserted at random positions in the first and second half.
"""
import hashlib
import random

from common import N_SETS, N_TRIALS

CHECKS = [  # (stimulus_id, instructed emotion, instructed strength)
    ("C01", "something_else", 2),
    ("C02", "excitement", 4),
]


def make_sets(stimuli):
    """stimuli: rows with stimulus_id and stratum (role == 'stimulus'), in stimuli.csv order."""
    rows = sorted(stimuli, key=lambda r: (r["stratum"], r["stimulus_id"]))
    sets = [[] for _ in range(N_SETS)]
    for i, r in enumerate(rows):
        sets[i % N_SETS].append(r["stimulus_id"])
    assert all(len(s) == N_TRIALS for s in sets), [len(s) for s in sets]
    return sets


def seed_for(participant_id: str) -> int:
    return int(hashlib.sha256(participant_id.encode()).hexdigest()[:16], 16)


def assign(index: int, participant_id: str, sets):
    """-> (set_index, plan). plan: list of trial dicts in presentation order (20 stimuli + 2 checks)."""
    rng = random.Random(seed_for(participant_id))
    k = index % N_SETS
    order = list(sets[k])
    rng.shuffle(order)
    plan = [{"kind": "stimulus", "stimulus_id": s, "ab_left": rng.choice(["r1", "r3"])} for s in order]
    half = len(plan) // 2
    p1 = rng.randint(4, half - 2)          # first check somewhere in trials 5-9
    p2 = rng.randint(half + 3, len(plan) - 2)  # second one in trials 14-19 (after the first insertion shifts)
    for pos, (sid, emo, strength) in zip((p1, p2), CHECKS):
        plan.insert(pos, {"kind": "check", "stimulus_id": sid, "expect_emotion": emo, "expect_strength": strength})
    return k, plan


def practice_trial():
    return {"kind": "practice", "stimulus_id": "P01", "ab_left": "r1"}
