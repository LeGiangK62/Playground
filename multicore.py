import os
import multiprocessing as mp
from utility_function import *
import mosek
import pickle

mosek.Env().putlicensepath("mosek.lic")

# Define multiple scenarios (num_UEs, num_APs)
all_scenarios = [
    [5, 50],
    [6, 30],
    [10, 30],
    [10, 50],
    [15, 50],
    [20, 50],
    [20, 80],
]

# Parameters
gamma_thr = 0.125  # 10 ** (-5/10)
nu = 1
power_receiver = 0.2
p_dl = 0.1  # / noise_p
num_samples_total = 150  # Total samples per scenario
num_SRs_list = [2]  # Number of secondary receivers


def generate_and_save(file_idx, num_samples, num_APs, num_UEs, num_SRs, gamma_thr, nu,
                      power_receiver, p_dl, folder, progress_queue):
    """
    Generate data using all_bench and save to a unique file.
    """
    file_name = f"{folder}/{num_APs}AP_{num_UEs}UE_{num_SRs}SR_{file_idx}.pkl"
    print(f"[START] Generating {num_samples} samples for {file_name}...")

    # Call generate_all_data to generate and save the data
    generate_all_data(num_samples, num_APs, num_UEs, num_SRs, gamma_thr, nu, power_receiver, p_dl, file_name)

    # Notify progress queue
    progress_queue.put(file_name)


def run_scenario_sequentially(all_scenarios, num_SRs_list, num_samples_total, gamma_thr, nu,
                              power_receiver, p_dl, folder="Data/17Mar"):
    """
    Runs multiple scenarios sequentially, utilizing all CPU cores for each scenario.
    """
    os.makedirs(folder, exist_ok=True)

    cpu_count = mp.cpu_count()  # Use all available CPU cores

    for num_SRs in num_SRs_list:
        for num_UEs, num_APs in all_scenarios:
            print(f"\n[INFO] Running scenario {num_APs} APs, {num_UEs} UEs, {num_SRs} SRs using {cpu_count} cores...")

            processes = []
            progress_queue = mp.Queue()

            # Divide samples equally among CPU cores
            num_files = cpu_count
            samples_per_file = num_samples_total // num_files
            remainder = num_samples_total % num_files  # Distribute remaining samples

            start_idx = 0  # File index for differentiation
            for i in range(num_files):
                file_samples = samples_per_file + (1 if i < remainder else 0)

                p = mp.Process(target=generate_and_save,
                               args=(start_idx, file_samples, num_APs, num_UEs, num_SRs, gamma_thr, nu,
                                     power_receiver, p_dl, folder, progress_queue))
                p.start()
                processes.append(p)
                start_idx += file_samples  # Increment file index

            # Monitor progress
            completed_files = 0
            while completed_files < num_files:
                file_done = progress_queue.get()
                completed_files += 1
                print(f"[DONE] {completed_files}/{num_files} -> {file_done}")

            # Wait for all processes to complete before moving to the next scenario
            for p in processes:
                p.join()

            print(f"[INFO] Scenario {num_APs} APs, {num_UEs} UEs, {num_SRs} SRs completed!\n")

    print("[INFO] All scenarios completed successfully!")


# Run sequential execution
if __name__ == "__main__":
    run_scenario_sequentially(all_scenarios, num_SRs_list, num_samples_total, gamma_thr, nu, power_receiver, p_dl)
