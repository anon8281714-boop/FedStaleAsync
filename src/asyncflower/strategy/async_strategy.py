from flwr.server.strategy import Strategy
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union, Literal, Tuple
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Scalar, Parameters

class AsyncStrategy(Strategy, ABC):
    @abstractmethod
    def round_delimiter(self, server_round: int) -> Tuple[Literal["buffer_size", "round_duration"], int|float]:
        pass

    @abstractmethod
    def aggregate_fit(
        self,
        server_round: int, 
        results: List[tuple[ClientProxy, FitRes]], 
        failures: List[Union[tuple[ClientProxy, FitRes], BaseException]],
        global_params: Optional[Parameters] = None
    ) -> tuple[Optional[Parameters], Dict[str, Scalar]]:
        pass