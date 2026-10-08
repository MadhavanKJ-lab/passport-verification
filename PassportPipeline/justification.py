# Generates a plain-language justification for a combined pipeline verdict,
# explaining which stage(s) drove the result and why, in terms a human reviewer
# can act on without reading the raw JSON or the combiner's math.

def _heuristics_explanation(heuristics_result):
    if heuristics_result is None:
        return 'Heuristics stage did not run.'
    flags = heuristics_result.get('confirmed_flags', [])
    if not flags:
        return (
            'No localized tampering flags — sharpness, error-level analysis, and '
            'noise-consistency checks found no region where 2+ independent signals '
            'agreed something looked pasted/altered. This does not confirm '
            'authenticity (see disclaimer), only that no obvious localized edit '
            'was detected.'
        )
    regions = ', '.join(f"({f['x']},{f['y']})" for f in flags)
    max_z = max(abs(f['zscore']) for f in flags)
    return (
        f'{len(flags)} localized region(s) flagged by 2+ independent checks at '
        f'pixel coordinates {regions} (strongest within-image z-score: {max_z:.2f}). '
        f'See heuristics_flags.png for a visual marker on each flagged tile.'
    )


def build_justification(combined_result, heuristics_result):
    raw = combined_result['stage_scores']
    weights = combined_result['weights']
    crop = combined_result['crop']

    lines = []
    lines.append(f"# Verdict: {combined_result['verdict'].upper()}")
    lines.append(f"Combined score: {combined_result['combined_score']:.3f} "
                 f"(inter-stage disagreement: {combined_result['disagreement']:.3f})")
    lines.append('')
    lines.append(
        'This is a weighted combination of three independent signals — each '
        'stage\'s OWN raw calibrated score, not re-normalized against a separate '
        'reference set (an earlier version did that and overfit to one narrow '
        'image source; fixed by trusting each model\'s own training-time '
        'calibration instead). TruFor is weighted highest as the most '
        'consistently validated signal in this project\'s own testing. A small '
        'bonus is added when stages disagree, so a strong single signal or real '
        'inter-stage conflict still surfaces for review rather than being '
        'averaged away — without flagging every image the way a cross-image '
        'baseline comparison previously did. NOT a trained/calibrated classifier '
        'and NOT a passport-authentication decision. Route elevated/high results '
        'to human review.'
    )
    lines.append('')

    lines.append('## Stage 1: Document crop')
    if crop.get('cropped'):
        lines.append(f"Document boundary found and cropped to {crop.get('output_size')}. "
                      f"See cropped.png.")
    else:
        lines.append(f"No confident document boundary found ({crop.get('reason')}) — "
                      f"original image used unchanged for all subsequent stages. This is "
                      f"common for images that are already tightly framed to the document.")
    lines.append('')

    lines.append('## Stage 2: Lenient tamper heuristics')
    lines.append(f"Weight in combined score: {weights['heuristics']:.0%}.")
    lines.append(_heuristics_explanation(heuristics_result))
    lines.append('')

    lines.append('## Stage 3: B-Free (synthetic/AI-generated image detector)')
    lines.append(f"Weight in combined score: {weights['bfree']:.0%}.")
    if raw['bfree_synthetic_probability'] is not None:
        lines.append(f"Raw synthetic-probability score: {raw['bfree_synthetic_probability']:.3f} "
                      f"(the model's own calibrated [0,1] output, not re-normalized).")
    else:
        lines.append('B-Free stage did not produce a usable result.')
    lines.append('')

    lines.append('## Stage 4: TruFor (general pixel-forensics)')
    lines.append(f"Weight in combined score: {weights['trufor']:.0%} (highest — see disclaimer).")
    if raw['trufor'] is not None:
        pooled = raw.get('trufor_pooled')
        localized = raw.get('trufor_localized')
        lines.append(f"Effective score used: {raw['trufor']:.3f} = max(pooled, localized).")
        lines.append(
            f"  - Pooled (whole-image) score: {pooled:.3f}" if pooled is not None
            else "  - Pooled (whole-image) score: n/a"
        )
        lines.append(
            f"  - Localized (connected-component concentration) score: {localized:.3f}" if localized is not None
            else "  - Localized score: n/a"
        )
        if localized is not None and pooled is not None and localized > pooled + 0.1:
            lines.append(
                '  This image\'s pooled score understates the risk — the anomaly map '
                'has a tight, concentrated high-confidence region (not diffuse edge '
                'noise) that the pooled whole-image average washes out. The localized '
                'score caught it instead.'
            )
        lines.append('See trufor_heatmap.png / trufor_overlay.png for the pixel-level '
                      'localization map, and trufor_confidence_map.png for where that map '
                      'should be trusted most.')
    else:
        lines.append('TruFor stage did not produce a usable result.')
    lines.append('')

    lines.append('## Which stage drove the verdict')
    risk_by_stage = {
        'TruFor': raw['trufor'],
        'B-Free': raw['bfree_synthetic_probability'],
        'Heuristics': raw['heuristics'],
    }
    risk_by_stage = {k: v for k, v in risk_by_stage.items() if v is not None}
    if risk_by_stage:
        top_stage = max(risk_by_stage, key=risk_by_stage.get)
        lines.append(f"Highest individual raw score: {top_stage} ({risk_by_stage[top_stage]:.3f}).")
        if combined_result['disagreement'] > 0.3:
            lines.append(
                f"Stages disagreed substantially (spread: {combined_result['disagreement']:.3f}) "
                f"— a bonus was added on top of the weighted average per this pipeline's "
                f"lean-toward-review-under-disagreement design."
            )

    return '\n'.join(lines)
