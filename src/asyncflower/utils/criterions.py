from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

class AvailableClientsCriterion(Criterion):
    def __init__(self, all_client_cids: list, busy_client_cids: list) -> None:
        self.available_client_cids = set(all_client_cids) - busy_client_cids

    def select(self, client: ClientProxy) -> bool:
        cid = client.cid
        return cid in self.available_client_cids