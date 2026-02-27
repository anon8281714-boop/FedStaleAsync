from asyncflower.utils import prob_estimator
from asyncflower.builders.participation_prob_estimation.estimation_method.client_group_based import build_client_group_based_estimator
from asyncflower.builders.participation_prob_estimation.estimation_method.client_group_lookup import build_client_group_lookup_estimator

def build_participation_prob_estimator(scheme: str, **params) -> prob_estimator.BaseParticipationProbabilityEstimator:
    if scheme == "scalar":
        return prob_estimator.ScalarParticipationProbabilityEstimator(
            probability = params["probability"]
        )
    
    if scheme == "uniform":
        return prob_estimator.UniformParticipationProbabilityEstimator(
            min_allowed_prob = params["min_allowed_prob"]
        )

    elif scheme == "maximum":
        return prob_estimator.MaximumParticipationProbabilityEstimator()

    elif scheme == "client_group_based":
        return build_client_group_based_estimator(**params)
    
    elif scheme == "client_group_lookup":
        return build_client_group_lookup_estimator(**params)

    elif scheme == "fedau_based":
        return prob_estimator.FedAUBasedParticipationProbabilityEstimator(
            participation_interval_cutoff = params["participation_interval_cutoff"]
        )

    raise ValueError(f"{scheme} is not a valid scheme.")