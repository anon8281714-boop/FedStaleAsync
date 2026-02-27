from typing import Literal, Callable
from asyncflower.builders.device_specifications.client_groups_exponential import build_client_groups_exponential
from asyncflower.builders.device_specifications.homogeneous_exponential import build_homogeneous_exponential

DeviceScheme = Literal["client_groups_exponential", "homogeneous_exponential"]

def build_devices(
    scheme: DeviceScheme = "client_groups_exponential",
    **kwargs
) -> str:
    if scheme == "client_groups_exponential":
        return build_client_groups_exponential(**kwargs)
    
    elif scheme == "homogeneous_exponential":
        return build_homogeneous_exponential(**kwargs)

    raise ValueError(f"{scheme} is not a valid scheme.")