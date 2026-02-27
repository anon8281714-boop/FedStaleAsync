from asyncflower.utils.prob_estimator import BaseParticipationProbabilityEstimator, SmoothedParticipationProbabilityEstimator
from asyncflower.builders.participation_prob_estimation.smooth_method.smoothed_vs_original_group_prob_ratio import build_smoothed_vs_original_group_prob_ratio_ratio

def build_smoothed_participation_prob_estimator(
    participation_prob_estimator: BaseParticipationProbabilityEstimator,
    scheme: str, 
    **params
) -> BaseParticipationProbabilityEstimator:
    """Generates an estimator that adds some value to the probabilities of participation estimated by `participation_prob_estimator`."""

    if scheme == "none":
        smooth_term_fn = lambda cid, prob: 0
    
    elif scheme == "smoothed_vs_original_prob_ratio":
        # smooth_term = min((ratio-1)*prob, 1-prob)
        assert params["ratio"] >= 1, "Ratio should be greater than 1."
        smooth_term_fn = lambda cid, prob: min((params["ratio"] - 1) * prob, 1 - prob)

    elif scheme == "smoothed_vs_original_group_prob_ratio":
        smooth_term_fn = build_smoothed_vs_original_group_prob_ratio_ratio(**params)

    else:
        raise ValueError(f"{scheme} is not a valid scheme.")
    
    return SmoothedParticipationProbabilityEstimator(
        participation_prob_estimator = participation_prob_estimator, 
        smooth_term_fn = smooth_term_fn
    )