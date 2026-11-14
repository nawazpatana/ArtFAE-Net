"""Analysis skeleton for the human-evaluation study. Not yet run on real data.

Inputs: trials.csv + participants.csv (export.py), study/stimuli.csv, study/private/votes.csv (ArtEmis shares),
study/precomputed/outputs.json (R1, R3).

0. Pilot data (phase = "pilot") are left out by default (`--pilot none`). `--pilot optin` keeps pilot participants
   who ticked future_use_optin, and may be used ONLY after a research ethics committee has approved that use.
   `--pilot all` is for synthetic smoke tests (pilot_sim.py) only.
1. Exclusion: participants who fail ANY of the 2 attention checks, or who did not finish, are excluded
   (decide and pre-register before looking at the data; `max_failed_checks` changes it). A SKIPPED attention
   check is not a pass, so it counts as failed. Practice and check trials never enter the analyses.
1b. Skips ("Skip this artwork") are missing data, not answers: the step it was pressed on and every later step of
   that trial are set to missing (mask_skips); steps submitted before the skip are kept. Rows of trials left
   unfinished (status = incomplete) only hold the submitted steps. Every metric uses only non-missing values.
2. New-human distribution per stimulus: the share of included participants choosing each emotion in the blind
   step (+ `alpha` pseudo-count per emotion when it is used as a target, so log-losses stay finite).
   Per stimulus, against R1 and R3: soft cross-entropy H(human, model), KL(human || model), top-1 agreement
   (model argmax in the set of human modal emotions, credited 1/k for a k-way human tie).
   Human ceiling: the ArtEmis vote shares scored the same way against the new humans.
   Reported as means over stimuli with a stimulus bootstrap 95% CI, and paired differences R1 - R3.
3. Plausibility (reveal step, 1-5): mean with a participant-cluster bootstrap 95% CI.
4. Preference (A/B step): R1 vs R3 choices, "no difference" excluded; exact two-sided binomial test, p = 0.5.
5. Every summary split by language group (the study UI language, and native language).

usage:  python study/analysis.py EXPORT_DIR [--out results.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
from common import EMOTIONS  # noqa: E402

B = 2000
RNG = 0


# ---------------------------------------------------------------- data
def load(export_dir: Path):
    t = pd.read_csv(export_dir / "trials.csv")
    p = pd.read_csv(export_dir / "participants.csv")
    votes = pd.read_csv(STUDY / "private/votes.csv").set_index("stimulus_id")
    out = json.loads((STUDY / "precomputed/outputs.json").read_text())["stimuli"]
    return t, p, votes, out


STEPS = ["blind", "reveal", "ab", "attn"]
STEP_FIELDS = {"blind": ["blind_emotion", "blind_strength", "blind_reason"],
               "reveal": ["plausibility"],
               "ab": ["ab_left_model", "ab_choice", "ab_choice_model"],
               "attn": ["attention_rating", "attention_skipped"]}


def _true(v) -> bool:
    return v is True or v in (1, "True", "true", "1")


def mask_skips(t: pd.DataFrame) -> pd.DataFrame:
    """Set the answers of the skipped step and every later step of a skipped trial to missing."""
    if "skipped" not in t:
        return t
    t = t.copy()
    sk = t.skipped.map(_true)
    at = t.skipped_step.map({s: i for i, s in enumerate(STEPS)})
    for i, step in enumerate(STEPS):
        rows = sk & (at <= i)
        cols = [c for c in STEP_FIELDS[step] if c in t]
        t[cols] = t[cols].astype(object)
        t.loc[rows, cols] = np.nan
    return t


def pilot_filter(p: pd.DataFrame, pilot: str = "none") -> set:
    """Participant ids allowed into the analysis given their phase (pilot data: see 0. above)."""
    phase = p["phase"].fillna(p["study_mode"]) if "phase" in p else p["study_mode"]
    is_pilot = phase.eq("pilot")
    if pilot == "all":
        keep = pd.Series(True, index=p.index)
    elif pilot == "optin":
        keep = ~is_pilot | p.get("future_use_optin", pd.Series(False, index=p.index)).map(_true)
    elif pilot == "none":
        keep = ~is_pilot
    else:
        raise ValueError(f"pilot must be none, optin or all, not {pilot!r}")
    return set(p.loc[keep, "participant_id"])


def included(t: pd.DataFrame, p: pd.DataFrame, max_failed_checks: int = 0, pilot: str = "none") -> pd.DataFrame:
    t = mask_skips(t)
    checks = t[t.kind == "check"].groupby("participant_id").check_pass.agg(lambda s: int((~s.map(_true)).sum()))
    done = set(p.loc[p.completed_at.notna(), "participant_id"]) & pilot_filter(p, pilot)
    ok = {pid for pid in done if checks.get(pid, 99) <= max_failed_checks}
    return t[(t.kind == "stimulus") & t.participant_id.isin(ok)]


# ---------------------------------------------------------------- metrics
def human_dist(trials: pd.DataFrame, alpha: float = 0.0) -> pd.DataFrame:
    c = pd.crosstab(trials.stimulus_id, trials.blind_emotion).reindex(columns=EMOTIONS, fill_value=0).astype(float)
    return (c + alpha).div((c + alpha).sum(axis=1), axis=0)


def soft_ce(h, q, eps=1e-12):
    return float(-(np.asarray(h) * np.log(np.clip(q, eps, 1))).sum())


def kl(h, q, eps=1e-12):
    h, q = np.asarray(h), np.clip(np.asarray(q), eps, 1)
    m = h > 0
    return float((h[m] * np.log(h[m] / q[m])).sum())


def top1(h, q):
    modes = np.flatnonzero(np.isclose(h, np.max(h)))
    return float(int(np.argmax(q)) in modes) / len(modes)  # 1/k credit for a k-way human tie


def per_stimulus(trials, votes, out, alpha=0.5):
    """One row per stimulus: n raters and the three scores for R1, R3 and the ArtEmis ceiling."""
    trials = trials.dropna(subset=["blind_emotion"])  # skipped at the blind step: missing
    raw, smooth = human_dist(trials), human_dist(trials, alpha)
    n = trials.groupby("stimulus_id").size()
    rows = []
    for sid in raw.index:
        h = raw.loc[sid].to_numpy()
        cands = {"r1": out[sid]["r1"], "r3": out[sid]["r3"],
                 "artemis": votes.loc[sid, [f"p_{e}" for e in EMOTIONS]].to_numpy(float)}
        r = {"stimulus_id": sid, "n": int(n[sid])}
        for k, q in cands.items():
            q = np.asarray(q, float)
            q_s = (q + 1e-3) / (q + 1e-3).sum() if k == "artemis" else q  # ArtEmis shares contain zeros
            r[f"softce_{k}"] = soft_ce(smooth.loc[sid].to_numpy(), q_s)
            r[f"kl_{k}"] = kl(smooth.loc[sid].to_numpy(), q_s)
            r[f"top1_{k}"] = top1(h, q)
        rows.append(r)
    return pd.DataFrame(rows)


def boot_mean(x, clusters=None, b=B, seed=RNG):
    """Mean with a percentile bootstrap 95% CI; resamples clusters (e.g. participants) when given."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(x), (b, len(x)))
        bs = x[idx].mean(1)
    else:
        groups = [x[np.asarray(clusters) == c] for c in pd.unique(clusters)]
        bs = np.array([np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
                       for _ in range(b)])
    return {"mean": float(x.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(len(x))}


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test (sum of outcomes no more likely than the observed one)."""
    if n == 0:
        return float("nan")
    pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return float(min(1.0, sum(q for q in pmf if q <= pmf[k] * (1 + 1e-9))))


def preference(trials):
    c = trials.ab_choice_model.dropna().value_counts()
    k, n3 = int(c.get("r1", 0)), int(c.get("r3", 0))
    return {"r1": k, "r3": n3, "same": int(c.get("same", 0)), "share_r1_of_decided": k / (k + n3) if k + n3 else None,
            "p_binomial": binom_two_sided(k, k + n3)}


def summarise(trials, votes, out):
    ps = per_stimulus(trials, votes, out)
    res = {"n_participants": int(trials.participant_id.nunique()), "n_trials": int(len(trials)),
           "n_stimuli": int(len(ps)), "per_stimulus_raters_median": float(ps.n.median()) if len(ps) else None}
    for m in ("softce", "kl", "top1"):
        for k in ("r1", "r3", "artemis"):
            res[f"{m}_{k}"] = boot_mean(ps[f"{m}_{k}"])
        res[f"{m}_r1_minus_r3"] = boot_mean(ps[f"{m}_r1"] - ps[f"{m}_r3"])
    pl = trials.dropna(subset=["plausibility"])
    res["plausibility"] = boot_mean(pl.plausibility.astype(float), clusters=pl.participant_id.to_numpy()) if len(pl) else None
    att = trials.dropna(subset=["attention_rating"])
    res["attention_rating"] = boot_mean(att.attention_rating.astype(float), clusters=att.participant_id.to_numpy()) if len(att) else None
    if "skipped" in trials:
        sk = trials[trials.skipped.map(_true)]
        res["skipped_trials"] = {"n": int(len(sk)), "by_step": {str(k): int(v) for k, v in sk.skipped_step.value_counts().items()}}
    res["preference"] = preference(trials)
    return res, ps


def analyse(export_dir: Path, max_failed_checks: int = 0, pilot: str = "none"):
    t, p, votes, out = load(export_dir)
    inc = included(t, p, max_failed_checks, pilot)
    res, ps = summarise(inc, votes, out)
    res["pilot_rule"] = pilot
    res["excluded_participants"] = int(p.participant_id.nunique() - inc.participant_id.nunique())
    native = p.set_index("participant_id").native_language
    res["by_ui_language"] = {g: summarise(d, votes, out)[0] for g, d in inc.groupby("lang")}
    res["by_native_language"] = {g: summarise(d, votes, out)[0]
                                 for g, d in inc.assign(native=inc.participant_id.map(native)).groupby("native")}
    return res, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--out")
    ap.add_argument("--max-failed-checks", type=int, default=0)
    ap.add_argument("--pilot", choices=["none", "optin", "all"], default="none",
                    help="pilot data: none (default); optin only after ethics approval; all for synthetic tests only")
    a = ap.parse_args()
    res, ps = analyse(Path(a.export_dir), a.max_failed_checks, a.pilot)
    txt = json.dumps(res, indent=1, default=float)
    if a.out:
        Path(a.out).write_text(txt)
        ps.to_csv(Path(a.out).with_suffix(".per_stimulus.csv"), index=False)
    print(txt[:3000])


if __name__ == "__main__":
    main()
