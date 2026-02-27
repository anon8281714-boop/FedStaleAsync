from typing import Dict 
from torch.distributions import Distribution

class DevicesSimulation:
    def __init__(self, service_time_distributions: Dict[str, Distribution]):
        self.service_time_distributions = service_time_distributions

    def sample_service_time(self, cid: str):
        return self.service_time_distributions[cid].sample().item()