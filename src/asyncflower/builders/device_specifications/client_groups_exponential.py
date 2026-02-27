import os
import joblib
import json
import numpy as np
import argparse
from torch.distributions import Exponential

def build_client_groups_exponential(
    num_clients: int,
    group_names: list[str],
    group_fractions: list[float],
    group_service_rates: list[float],
    save_dir: str,
    force_sort_groups: bool = True,
) -> str:
    if force_sort_groups:
        group_idx_order = np.argsort(group_names)
        group_names = np.array(group_names)[group_idx_order].tolist()
        group_fractions = np.array(group_fractions)[group_idx_order]
        group_service_rates = np.array(group_service_rates)[group_idx_order]

    setting_str_id = f"n{num_clients}"
    for name, fraction, service_rate in zip(group_names, group_fractions, group_service_rates):
        fraction = str(fraction).replace('.', 'd')
        service_rate = str(service_rate).replace('.', 'd')
        setting_str_id += f"-{name}_f{fraction}_exp{service_rate}"

    devices_specification_dir = f"{save_dir}/device/{setting_str_id}"

    if not os.path.exists(devices_specification_dir):
        os.makedirs(devices_specification_dir, exist_ok = True)

        assert sum(group_fractions) == 1, "Group fractions should sum 1."
        
        group_total_clients = np.floor(np.array(group_fractions) * num_clients).astype(int)
        remaining = num_clients - group_total_clients.sum().item()
        group_total_clients[:remaining] += 1

        group_service_time_distributions = [Exponential(service_rate) for service_rate in group_service_rates]

        groups_dict = {}
        next_client_idx = 0
        for name, n_clients, distribution in zip(group_names, group_total_clients, group_service_time_distributions):
            groups_dict[name] = {
                "cid_list": [str(cid) for cid in range(next_client_idx, next_client_idx + n_clients)], 
                "service_time_distribution": distribution
            } 
            next_client_idx = next_client_idx + n_clients

        service_time_distributions = {
            cid: group["service_time_distribution"] for _, group in groups_dict.items() for cid in group["cid_list"]
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
    parser.add_argument("--group_names", required = True, dest = "group_names", nargs = "+", type = str)
    parser.add_argument("--group_fractions", required = True, dest = "group_fractions", nargs = "+", type = float)
    parser.add_argument("--group_service_rates", required = True, dest = "group_service_rates", nargs = "+", type = float)
    
    args = parser.parse_args()

    print("Preparing Devices...")
    devices_specification_dir = build_client_groups_exponential(
        num_clients = args.num_clients,
        group_names = args.group_names,
        group_fractions = args.group_fractions,
        group_service_rates = args.group_service_rates,
        save_dir = args.save_dir
    )
    print(f"Device specifications were stored in {devices_specification_dir}...")