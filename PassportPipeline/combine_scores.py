# Combine TruFor + B-Free + heuristics scores into one risk verdict.
#
# This is a transparent RULE-BASED combiner, not a trained classifier. There is no
# labeled real-vs-forged dataset to fit a logistic regression / XGBoost model on
# responsibly — fitting one anyway would be curve-fitting noise and reporting
# invented confidence.
#
# IMPORTANT, learned the hard way: an earlier version of this combiner z-scored
# each stage's raw output against a small "known-clean" reference set built from
# only one narrow image source (a 10-image Kaggle synthetic batch). That caused
# every image from ANY other source to read as a statistical outlier and get
# flagged "high" — it was detecting "doesn't look like the Kaggle batch," not
# forgery. Fixed by dropping cross-image baselining entirely: TruFor's and
# B-Free's raw [0,1] outputs are already calibrated by their own authors against
# far larger and more diverse datasets than anything we could assemble ourselves,
# so we trust those scales directly instead of re-normalizing against our own
# tiny sample.
#
# Weighted combination (not pure max()): TruFor gets the highest weight as the
# most consistently validated signal this session (sensible, non-degenerate
# scores across a real 15-image batch, architecture purpose-built for local
# tampering detection). B-Free is weighted lower — useful complementary signal,
# but showed real domain-gap behavior this session (e.g. reading "real" on
# content literally named synthetic/fake). Heuristics lowest — narrowest scope by
# design, rarely fires, acts mainly as a tie-breaker/corroborating signal.
#
# A small, BOUNDED disagreement bonus (not z-score-based, so it cannot blow up
# the way the previous version did) still lets strong single-stage signals or
# real inter-stage conflict push the verdict up — "lean toward forged" without
# flagging every image, which is the explicit balance this version is tuned for.
#
# TruFor's own pooled trufor_score averages evidence across the WHOLE image, so a
# small, surgical edit (e.g. a couple of added characters) in an otherwise large
# pristine document barely moves it — confirmed on a real edited passport image
# where trufor_score read 0.078 despite a 0.997-confidence anomaly-map region
# exactly at the edit. passport_trufor.py now also reports
# 'trufor_localized_score' (connected-component concentration analysis of the
# anomaly map — validated to distinguish a genuine tight tampered blob from the
# scattered natural-edge noise clean images also show at their peak pixels,
# which raw peak/percentile values alone could NOT do). The effective TruFor
# signal fed into this combiner is max(trufor_score, trufor_localized_score), so
# a tiny but highly-localized edit isn't washed out by the rest of the document.
#
# If real labeled data becomes available later, replace the body of combine()
# with a loaded model's .predict_proba() call — nothing else in the pipeline
# needs to change.

WEIGHTS = {
    'trufor': 0.5,
    'bfree': 0.3,
    'heuristics': 0.2,
}

DISAGREEMENT_WEIGHT = 0.1  # bonus scaled by (max - min) of the available raw
                            # [0,1] scores — bounded by construction, unlike a
                            # z-score-based bonus which has no natural ceiling

HIGH_THRESHOLD = 0.55
ELEVATED_THRESHOLD = 0.30
# Deliberately lower than each individual stage's own 0.33/0.66 risk_label
# convention on the high end is close to TruFor's own "high" cut, but the
# elevated/review bar sits below it so a single strong signal or real
# disagreement still surfaces for human review rather than being averaged away.


def normalize_heuristics_score(heuristics_result):
    """
    heuristics_result is the dict returned by heuristics.run_heuristics(). Its
    z-score is already computed PER-IMAGE (a flagged tile vs. that same image's
    own other tiles) — that's a legitimate, self-contained normalization, not the
    cross-image baselining that caused the earlier bug, so it needs no further
    adjustment here.
    """
    if heuristics_result is None:
        return None
    flags = heuristics_result.get('confirmed_flags', [])
    if not flags:
        return 0.0
    max_abs_z = max(abs(f['zscore']) for f in flags)
    return min(1.0, max_abs_z / 10.0)


def combine(trufor_score, bfree_score, heuristics_score):
    """All three inputs are the stages' own raw [0,1] scores (or None if that
    stage didn't run)."""
    available = {
        'trufor': trufor_score,
        'bfree': bfree_score,
        'heuristics': heuristics_score,
    }
    available = {k: v for k, v in available.items() if v is not None}
    if not available:
        return 0.0, 0.0

    total_weight = sum(WEIGHTS[k] for k in available)
    weighted_avg = sum(WEIGHTS[k] * v for k, v in available.items()) / total_weight

    disagreement = max(available.values()) - min(available.values())
    final_score = min(1.0, weighted_avg + DISAGREEMENT_WEIGHT * disagreement)

    return final_score, disagreement


def bucket(final_score):
    if final_score >= HIGH_THRESHOLD:
        return 'high'
    if final_score >= ELEVATED_THRESHOLD:
        return 'elevated — review recommended'
    return 'low'


def build_combined_result(image_path, crop_info, trufor_result, bfree_result, heuristics_result):
    """
    trufor_result / bfree_result: the JSON dicts produced by passport_trufor.py /
    passport_bfree.py (or None if that stage failed to run).
    heuristics_result: the dict returned by heuristics.run_heuristics().
    """
    trufor_pooled = trufor_result.get('trufor_score') if trufor_result else None
    trufor_localized = trufor_result.get('trufor_localized_score') if trufor_result else None
    # effective TruFor signal: whichever is higher, pooled whole-image score or
    # the localized-concentration score — see module docstring for why neither
    # alone is sufficient (pooled score washes out tiny edits; localized score
    # alone was found to also need the pooled score as a floor, since it reads
    # as exactly 0.0 when there's no anomaly at all, same as a genuinely clean
    # image, so taking the max of both costs nothing and only adds coverage)
    trufor_raw = None
    if trufor_pooled is not None or trufor_localized is not None:
        trufor_raw = max(trufor_pooled or 0.0, trufor_localized or 0.0)

    bfree_raw = bfree_result.get('synthetic_probability') if bfree_result else None
    heuristics_raw = normalize_heuristics_score(heuristics_result)

    final_score, disagreement = combine(trufor_raw, bfree_raw, heuristics_raw)

    # A crashed detector must never make an image look clean: if a model stage
    # failed, the remaining stages can still raise the verdict, but they can't
    # clear the image on their own.
    failed_stages = [name for name, val in (('trufor', trufor_raw), ('bfree', bfree_raw)) if val is None]
    verdict = bucket(final_score)
    if failed_stages and verdict == 'low':
        verdict = 'cannot assess — review required'

    return {
        'image': image_path,
        'crop': crop_info,
        'stage_scores': {
            'trufor': trufor_raw,
            'trufor_pooled': trufor_pooled,
            'trufor_localized': trufor_localized,
            'bfree_synthetic_probability': bfree_raw,
            'heuristics': heuristics_raw,
        },
        'weights': dict(WEIGHTS),
        'combined_score': final_score,
        'verdict': verdict,
        'failed_stages': failed_stages,
        'disagreement': disagreement,
        'disclaimer': (
            'Rule-based weighted combination of each stage\'s own calibrated score '
            '(TruFor 50%, B-Free 30%, heuristics 20%) plus a small bonus when '
            'stages disagree — not a trained/calibrated classifier. No labeled '
            'ground-truth data exists yet to fit or validate one. TruFor is '
            'weighted highest as the most consistently validated signal in this '
            'project\'s own testing; weights and thresholds are starting points, '
            'not a validated calibration. "trufor" is max(trufor_pooled, '
            'trufor_localized) — the pooled whole-image score and a connected-'
            'component concentration score that catches small localized edits the '
            'pooled score alone washes out. Not a passport-authentication decision '
            '— route elevated/high results to human review.'
        ),
    }
