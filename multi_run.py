import os
from utility_function import *
import multiprocessing as mp
import mosek


mosek.Env().putlicensepath("mosek.lic")


def generate_and_save(file_idx, num_samples, num_APs, num_UEs, num_SRs, gamma_thr, nu,
                      power_receiver, p_dl, folder, progress_queue):
    """
    Generate data using all_bench and save to a unique file.
    """
    file_name = f"{folder}/{num_APs}AP_{num_UEs}UE_{num_SRs}SR_{file_idx}.pkl"
    print(f"[START] Generating {num_samples} samples for {file_name}...")

    # Call all_bench to generate and save the data directly
    all_bench(num_samples, num_APs, num_UEs, num_SRs, gamma_thr, nu, power_receiver, p_dl, file_name)

    # Notify the main process that this file is done
    progress_queue.put(file_name)


def parallel_generate(num_samples_total, num_APs, num_UEs, num_SRs, gamma_thr, nu,
                      power_receiver, p_dl, folder="Data/12Mar"):
    """
    Splits the total number of samples among several files and uses multiprocessing
    to generate and save them in parallel.
    """
    os.makedirs(folder, exist_ok=True)

    # Get CPU cores and determine the number of files dynamically
    cpu_count = mp.cpu_count()
    num_files = min(cpu_count, num_samples_total)  # Ensure we don't create more files than samples
    samples_per_file = num_samples_total // num_files
    remainder = num_samples_total % num_files

    processes = []
    progress_queue = mp.Queue()

    print(f"[INFO] Using {num_files} files (based on {cpu_count} CPU cores) to generate data...")

    start_idx = 0  # This index differentiates filenames
    for i in range(num_files):
        # Distribute any extra samples to balance the load
        file_samples = samples_per_file + (1 if i < remainder else 0)
        p = mp.Process(target=generate_and_save,
                       args=(start_idx, file_samples, num_APs, num_UEs, num_SRs, gamma_thr, nu,
                             power_receiver, p_dl, folder, progress_queue))
        p.start()
        processes.append(p)
        start_idx += file_samples  # Increment file index (used in file naming)

    # Monitor progress from child processes
    completed_files = 0
    while completed_files < num_files:
        file_done = progress_queue.get()
        completed_files += 1
        print(f"[DONE] {completed_files}/{num_files} -> {file_done}")

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("[INFO] All files generated successfully!")


# Example usage:
if __name__ == "__main__":
    # Simulation parameters
    gamma_thr = 0.125  # 10 ** (-5/10)
    nu = 1
    power_receiver = 0.2
    p_dl = 0.1  # / noise_p
    num_APs = 20
    num_UEs = 2
    num_SRs = 2
    num_samples_total = 500  # Total samples to generate

    parallel_generate(num_samples_total, num_APs, num_UEs, num_SRs, gamma_thr, nu,
                      power_receiver, p_dl)
