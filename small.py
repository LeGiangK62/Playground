import numpy as np
from utility_function import *
import mosek


mosek.Env().putlicensepath("mosek.lic")

if __name__ == '__main__':
    # Define parameter ranges
    num_samples = 500
    num_UEs_list = [5, 10, 15]
    num_APs_list = [30, 45, 60]
    num_SRs_list = [2]

    all_scenarios = [
        # [6, 30],
        # [10, 30],
        # [10, 50],
        [20, 80],
        # [5, 50],
        # [10, 50],
        # [15, 50],
        # [20, 50],
        # [25, 50],
    ]
    gamma_thr = 0.125  # 10 ** (-5/10)
    nu = 1
    power_receiver = 0.2
    p_dl = 0.1  # / noise_p

    # Loop through all combinations of num_UEs and num_APs
    for num_SRs in num_SRs_list:
        for num_UEs, num_APs in all_scenarios:
            data_file_name = f'Data/11Mar/eval_{num_APs}AP_{num_UEs}UE_{num_SRs}SR.pkl'
            print(f"Running all_bench for {num_APs} APs, {num_UEs} UEs, and {num_SRs} SRs...")
            # Call function
            all_bench(num_samples, num_APs, num_UEs, num_SRs, gamma_thr, nu, power_receiver, p_dl, data_file_name)
            # print(f"Finished {num_APs} APs, {num_UEs} UEs, and {num_SRs} SRs. Data saved to {data_file_name}.\n")
