from abc import ABC, abstractmethod
from typing import Callable

class BaseParticipationProbabilityEstimator(ABC):
    @abstractmethod
    def update_estimations(self, server_round: int, participating_cids: list[int]):
        pass

    @abstractmethod
    def get_participation_probability(self, cid: int) -> float:
        pass

class FedAUBasedParticipationProbabilityEstimator(BaseParticipationProbabilityEstimator):
    # This participation probability estimation is inspired on the approach proposed by the FedAU method
    def __init__(self, participation_interval_cutoff: int):
        self.participation_interval_cutoff = participation_interval_cutoff
        self._client_participation_stats = {}

    def update_estimations(self, server_round: int, participating_cids: list[int]):
        for cid in participating_cids:        
            cid = int(cid)
            self._client_participation_stats[cid] = self._client_participation_stats.get(
                cid, {"num_participations": 0, "last_interval_estimation": 0, "last_participation_round": -1}
            )
            self._client_participation_stats[cid]["num_participations"] += 1 

            num_participations = self._client_participation_stats[cid]["num_participations"]
            last_interval_estimation = self._client_participation_stats[cid]["last_interval_estimation"]
            last_participation_round = self._client_participation_stats[cid]["last_participation_round"]
            
            cur_no_participation_interval = min(
                server_round - last_participation_round, self.participation_interval_cutoff
            )
            no_participation_interval_estimation = ((num_participations - 1) * last_interval_estimation + cur_no_participation_interval) / num_participations
            self._client_participation_stats[cid]["participation_prob"] = 1/no_participation_interval_estimation

            self._client_participation_stats[cid]["last_interval_estimation"] = no_participation_interval_estimation 
            self._client_participation_stats[cid]["last_participation_round"] = server_round 

    def get_participation_probability(self, cid):
        prob = self._client_participation_stats.get(cid, {"participation_prob": 1/self.participation_interval_cutoff})["participation_prob"]
        # TODO: Remove the print below after debug.
        print(f"CID: {cid}, Prob: {prob}")
        return prob
    
class ClientGroupBasedParticipationProbabilityEstimator(BaseParticipationProbabilityEstimator):
    # First, we estimate individual participation probabilities by dividing the number of contributions by the number of rounds. Then, we estimage the probability of participation for each group by taking the average between the individual probabilities of the group's members. These group-level probability estimations are then used for the clients. 

    def __init__(self, cid_to_group: dict, min_allowed_prob: float | None = None):
        self.cid_to_group = cid_to_group
        self.min_allowed_prob = min_allowed_prob if min_allowed_prob else 0
        self._client_participation_stats = {}
        self._group_probs = {}

    def update_estimations(self, server_round: int, participating_cids: list[int]):
        for cid in participating_cids:
            cid = int(cid)
            cid_group = self.cid_to_group[cid]

            self._client_participation_stats[cid] = self._client_participation_stats.get(
                cid, {"num_participations": 0}
            )
            self._client_participation_stats[cid]["num_participations"] += 1 

            self._client_participation_stats[cid]["participation_prob"] = self._client_participation_stats[cid]["num_participations"] / (server_round + 1)

        group_probs_sum = {}
        group_total_clients = {}

        for cid in self._client_participation_stats.keys():
            cid_group = self.cid_to_group[cid]
            prob = self._client_participation_stats[cid]["participation_prob"]
            group_probs_sum[cid_group] = group_probs_sum.get(cid_group, 0)
            group_probs_sum[cid_group] += prob

            group_total_clients[cid_group] = group_total_clients.get(cid_group, 0)
            group_total_clients[cid_group] += 1
    
        self._group_probs = {group: prob_sum / group_total_clients[group] for group, prob_sum in group_probs_sum.items()}

    def get_participation_probability(self, cid):
        cid = int(cid)
        cid_group = self.cid_to_group[cid]
        prob_before_clipping = self._group_probs.get(cid_group, self.min_allowed_prob)
        prob_after_clipping = max(prob_before_clipping, self.min_allowed_prob)
        # TODO: Remove the print below after debug.
        print(f"CID: {cid}, Prob. before clipping: {prob_before_clipping}, Prob. after clipping: {prob_after_clipping}")
        return prob_after_clipping

class UniformParticipationProbabilityEstimator(BaseParticipationProbabilityEstimator):
    # Uniform probabilities of participation. It considers the total number of clients and the buffer size.
    def __init__(self, min_allowed_prob: float | None = None):
        self.min_allowed_prob = min_allowed_prob if min_allowed_prob else 0
        self._all_known_cids = set()
        self._participation_prob = None

    def update_estimations(self, server_round: int, participating_cids: list[int]):
        self._all_known_cids = self._all_known_cids.union(set(participating_cids))
        total_participating_clients = len(participating_cids)
        total_known_cids = len(self._all_known_cids)
        self._participation_prob = total_participating_clients / total_known_cids

    def get_participation_probability(self, cid):
        prob_before_clipping = self._participation_prob
        prob_after_clipping = max(prob_before_clipping, self.min_allowed_prob)
        # TODO: Remove the print below after debug.
        print(f"CID: {cid}, Prob. before clipping: {prob_before_clipping}, Prob. after clipping: {prob_after_clipping}")
        return prob_after_clipping
    
class MaximumParticipationProbabilityEstimator(BaseParticipationProbabilityEstimator):
    # Probabilities of participation equals to 1.
    def __init__(self):
        pass
        
    def update_estimations(self, server_round: int, participating_cids: list[int]):
        pass

    def get_participation_probability(self, cid):
        # TODO: Remove the print below after debug.
        prob = 1
        print(f"CID: {cid}, Prob: {prob}")
        return prob
    
class ScalarParticipationProbabilityEstimator(BaseParticipationProbabilityEstimator):
    # Probabilities of participation equals to `probability`.
    def __init__(self, probability):
        self.probability = probability
        
    def update_estimations(self, server_round: int, participating_cids: list[int]):
        pass

    def get_participation_probability(self, cid):
        # TODO: Remove the print below after debug.
        print(f"CID: {cid}, Prob: {self.probability}")
        return self.probability
    
class LookupParticipationProbabilityEstimator(BaseParticipationProbabilityEstimator):
    # Probabilities of participation are given.
    def __init__(self, cid_probabilities: dict, min_allowed_prob: float | None = None):
        self.min_allowed_prob = min_allowed_prob if min_allowed_prob else 0
        self.cid_probabilities = cid_probabilities

    def update_estimations(self, server_round: int, participating_cids: list[int]):
        pass

    def get_participation_probability(self, cid):
        cid = int(cid)
        prob_before_clipping = self.cid_probabilities[cid]
        prob_after_clipping = max(prob_before_clipping, self.min_allowed_prob)
        # TODO: Remove the print below after debug.
        print(f"CID: {cid}, Prob. before clipping: {prob_before_clipping}, Prob. after clipping: {prob_after_clipping}")
        return prob_after_clipping
    
class SmoothedParticipationProbabilityEstimator(BaseParticipationProbabilityEstimator):
    # This implementation can be used to balance the tradeoff between unbiased aggregation and low variance
    def __init__(
        self, 
        participation_prob_estimator: BaseParticipationProbabilityEstimator,
        smooth_term_fn: Callable[[int, float], float] = None
    ):
        self.participation_prob_estimator = participation_prob_estimator
        self.smooth_term_fn = smooth_term_fn
    
    def update_estimations(self, server_round: int, participating_cids: list[int]):
        self.participation_prob_estimator.update_estimations(
            server_round = server_round, participating_cids = participating_cids
        )

    def get_participation_probability(self, cid):
        smoothed_prob = self.participation_prob_estimator.get_participation_probability(cid)
        if self.smooth_term_fn: 
            smoothed_prob += self.smooth_term_fn(cid = cid, prob = smoothed_prob)
        return smoothed_prob