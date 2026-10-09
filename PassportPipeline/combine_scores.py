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
# signal fed into this combiner is max(trufor_score, noise-floored
# trufor_localized_score) (see below), so a tiny but highly-localized edit isn't
# washed out by the rest of the document.
#
# Noise floors (second revision, also learned the hard way): feeding raw scores
# straight into the average made genuine phone photos of real passports read
# "elevated". Across the saved VerificationResults:
#   - B-Free reads 0.58-0.60 on real phone photos, 0.95-1.0 on AI-generated
#     documents, and 0.18-0.36 on locally edited fakes (not what it detects).
#     Its mid-range is camera/JPEG domain gap, not evidence, so only the part
#     above 0.5 counts, rescaled to [0,1].
#   - trufor_localized_score is a concentration ratio, not a probability. Clean
#     images sit at 0.0-0.30; real localized edits hit 1.0. Only the part above
#     0.4 counts, rescaled to [0,1].
#   - The disagreement bonus used max-min over all three stages, but heuristics
#     is 0.0 on nearly every image, so the "bonus" was really 0.1 * max score
#     and always pushed upward. Disagreement is now measured between the two
#     model stages only.
# Because each stage detects a different thing (local edits vs AI generation vs
# pasted regions), averaging dilutes a single real signal. Floors keep that
# signal visible: any stage in its own "medium" band forces at least elevated,
# a very strong model score forces high, and any confirmed heuristics flag
# forces at least elevated.
# These remap points come from only 3 real and 8 fake labeled images. They are
# starting values read off observed data, not a calibration.
#
# If real labeled data becomes available later, replace the body of combine()
# with a loaded model's .predict_proba() call — nothing else in the pipeline
# needs to change.

WEIGHTS = {
    'trufor': 0.5,
    'bfree': 0.3,
    'heuristics': 0.2,
}

DISAGREEMENT_WEIGHT = 0.1  # bonus scaled by |trufor - bfree| effective scores —
                            # bounded by construction, unlike a z-score-based
                            # bonus which has no natural ceiling

BFREE_NOISE_FLOOR = 0.5       # synthetic_probability at/below this counts as 0
LOCALIZED_NOISE_FLOOR = 0.4   # trufor_localized_score at/below this counts as 0

MEDIUM_STAGE_SCORE = 0.33     # any effective stage score >= this -> at least elevated
STRONG_STAGE_SCORE = 0.85     # any effective model score >= this -> at least high

HIGH_THRESHOLD = 0.55
ELEVATED_THRESHOLD = 0.30
# Deliberately lower than each individual stage's own 0.33/0.66 risk_label
# convention on the high end is close to TruFor's own "high" cut, but the
# elevated/review bar sits below it so a single strong signal or real
# disagreement still surfaces for human review rather than being averaged away.


def _rescale_above(value, floor):
    """Map [floor, 1] onto [0, 1]; anything at or below floor becomes 0."""
    if value is None:
        return None
    return max(0.0, min(1.0, (value - floor) / (1.0 - floor)))


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
    """All three inputs are effective [0,1] scores, already noise-floored (or
    None if that stage didn't run). Returns (final_score, disagreement,
    floor_applied)."""
    available = {
        'trufor': trufor_score,
        'bfree': bfree_score,
        'heuristics': heuristics_score,
    }
    available = {k: v for k, v in available.items() if v is not None}
    if not available:
        return 0.0, 0.0, None

    total_weight = sum(WEIGHTS[k] for k in available)
    weighted_avg = sum(WEIGHTS[k] * v for k, v in available.items()) / total_weight

    # heuristics is excluded: it is 0.0 on nearly every image, so including it
    # turned max-min into "highest score" and inflated every result
    models = [v for k, v in available.items() if k != 'heuristics']
    disagreement = max(models) - min(models) if len(models) == 2 else 0.0
    final_score = min(1.0, weighted_avg + DISAGREEMENT_WEIGHT * disagreement)

    floor_applied = None
    if max(available.values()) >= MEDIUM_STAGE_SCORE and final_score < ELEVATED_THRESHOLD:
        final_score, floor_applied = ELEVATED_THRESHOLD, 'medium_stage_score'
    if models and max(models) >= STRONG_STAGE_SCORE and final_score < HIGH_THRESHOLD:
        final_score, floor_applied = HIGH_THRESHOLD, 'strong_model_score'
    if available.get('heuristics', 0.0) > 0 and final_score < ELEVATED_THRESHOLD:
        final_score, floor_applied = ELEVATED_THRESHOLD, 'heuristics_flag'

    return final_score, disagreement, floor_applied


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
    trufor_eff = None
    if trufor_pooled is not None or trufor_localized is not None:
        trufor_raw = max(trufor_pooled or 0.0, trufor_localized or 0.0)
        # pooled is TruFor's own calibrated probability and is used as-is; the
        # localized concentration ratio only counts above its noise floor
        trufor_eff = max(trufor_pooled or 0.0,
                         _rescale_above(trufor_localized, LOCALIZED_NOISE_FLOOR) or 0.0)

    bfree_raw = bfree_result.get('synthetic_probability') if bfree_result else None
    bfree_eff = _rescale_above(bfree_raw, BFREE_NOISE_FLOOR)
    heuristics_raw = normalize_heuristics_score(heuristics_result)

    final_score, disagreement, floor_applied = combine(trufor_eff, bfree_eff, heuristics_raw)

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
        'effective_scores': {
            'trufor': trufor_eff,
            'bfree': bfree_eff,
            'heuristics': heuristics_raw,
        },
        'weights': dict(WEIGHTS),
        'combined_score': final_score,
        'verdict': verdict,
        'failed_stages': failed_stages,
        'disagreement': disagreement,
        'floor_applied': floor_applied,
        'disclaimer': (
            'Rule-based weighted combination (TruFor 50%, B-Free 30%, heuristics '
            '20%) of noise-floored stage scores, plus a small bonus when the two '
            'model stages disagree — not a trained/calibrated classifier. B-Free '
            f'only counts above {BFREE_NOISE_FLOOR} and TruFor\'s localized score '
            f'only above {LOCALIZED_NOISE_FLOOR} (both rescaled to [0,1]), since '
            'real photos routinely score below those. Floors keep a single strong '
            'stage from being averaged away: any stage in its own medium band '
            'forces at least elevated, a very strong model score forces high, and '
            'a confirmed heuristics flag forces at least elevated. Weights, noise '
            'floors and thresholds are starting points read off a small labeled '
            'sample (3 real, 8 fake), not a validated calibration. Not a '
            'passport-authentication decision — route elevated/high results to '
            'human review.'
        ),
    }
