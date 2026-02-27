import numpy as np
from typing import Callable
from flwr.common import FitRes, Metrics, parameters_to_ndarrays

def build_update_metrics_enrichment_fn(
    compute_update_norm: bool = False
) -> Callable[[FitRes], Metrics]:
    def metrics_enrichment_fn(update_fit_res: FitRes) -> Callable[[FitRes], Metrics]:
        metrics_dict = {}
        
        if compute_update_norm:
            ndarrays = parameters_to_ndarrays(update_fit_res.parameters)
            l2_norm = np.sqrt(sum(np.sum(layer_params ** 2) for layer_params in ndarrays)).item()
            metrics_dict["norm"] = l2_norm
        
        return metrics_dict

    return metrics_enrichment_fn