import os
import joblib
import json
import argparse
from torch.distributions import Exponential

def build_homogeneous_exponential(
    num_clients: int,
    service_rate: float,
    save_dir: str
) -> str:
    service_rate_str = str(service_rate).replace(".", "d")
    setting_str_id = f"n{num_clients}-exp{service_rate_str}"
    
    devices_specification_dir = f"{save_dir}/device/{setting_str_id}"

    if not os.path.exists(devices_specification_dir):
        os.makedirs(devices_specification_dir, exist_ok = True)

        service_time_distributions = {
            str(cid): Exponential(service_rate) for cid in range(num_clients)
        }

        joblib.dump(service_time_distributions, f"{devices_specification_dir}/specifications.joblib")

        with open(f"{devices_specification_dir}/specifications.json", "w") as f:
            service_time_distributions_json = {cid: distribution.__str__() for cid, distribution in service_time_distributions.items()}
            json.dump(service_time_distributions_json, f)

    return devices_specification_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_clients", required = True, dest = "num_clients", type = int)
    parser.add_argument("--save_dir", required = True, dest = "save_dir", type = str)
    parser.add_argument("--service_rate", required = True, dest = "service_rate", type = float)
    
    args = parser.parse_args()

    print("Preparing Devices...")
    devices_specification_dir = build_homogeneous_exponential(
        num_clients = args.num_clients,
        service_rate = args.service_rate,
        save_dir = args.save_dir
    )
    print(f"Device specifications were stored in {devices_specification_dir}...")