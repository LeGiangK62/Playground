import pickle
import numpy as np


def load_data_eval(file_name):
    with open(file_name, "rb") as f:
        data = pickle.load(f)

    # Extract data from the dictionary
    varsigma = data.get("varsigma", None)
    v = data.get("v", None)
    RCS = data.get("RCS", None)
    AP_loc = data.get("AP_loc", None)
    SR_loc = data.get("SR_loc", None)
    tar_loc = data.get("tar_loc", None)
    Pd = data.get("Pd", None)
    q_a = data.get("q_a", None)
    q_b = data.get("q_b", None)
    q_c = data.get("q_c", None)
    power_dl_sol_relaxed = data.get("power_dl_sol_relaxed", None)
    power_dl_sol_decoupled = data.get("power_dl_sol_decoupled", None)
    ap_act_sol_relaxed = data.get("ap_act_relaxed_sol", None)
    ap_act_sol_decoupled = data.get("ap_act_decoupled_sol", None)
    power_receiver = data.get("power_receiver", None)
    gamma_thr = data.get("gamma_thr", None)
    sum_power_relaxed = data.get("sum_power_relaxed", None)
    sum_power_relaxed_full = data.get("sum_power_relaxed", None)
    sum_power_decoupled = data.get("sum_power_decoupled", None)
    avg_time_relaxed = data.get("avg_time_relaxed", None)
    avg_time_decoupled = data.get("avg_time_decoupled", None)

    return (
        varsigma, v, RCS, AP_loc, SR_loc, tar_loc, Pd, q_a, q_b, q_c,
        power_receiver, gamma_thr, sum_power_relaxed, sum_power_relaxed_full, sum_power_decoupled,
        ap_act_sol_relaxed, ap_act_sol_decoupled,
        power_dl_sol_relaxed, power_dl_sol_decoupled, avg_time_relaxed, avg_time_decoupled
    )


if __name__ == '__main__':
    folder = "Data/17Mar"  # Adjust to your actual folder path
    output_file = f"{folder}/eval_10AP_2UE_2S.pkl"
    varsigma, v, RCS, AP_loc, SR_loc, tar_loc, Pd, q_a, q_b, q_c, power_receiver, gamma_thr, \
        sum_power_relaxed, sum_power_relaxed_full, sum_power_decoupled, ap_act_sol_relaxed, ap_act_sol_decoupled, \
        power_dl_sol_relaxed, power_dl_sol_decoupled, avg_time_relaxed, avg_time_decoupled = load_data_eval(output_file)

    print(f'varsigma: {varsigma.shape}')
    print(f'v: {v.shape}')
    print(f'RCS: {RCS.shape}')
    print(f'AP_loc: {AP_loc.shape}')
    print(f'SR_loc: {SR_loc.shape}')
    print(f'tar_loc: {tar_loc.shape}')
    print(f'Pd: {Pd}')
    print(f'q_a: {q_a.shape}')
    print(f'q_b: {q_b.shape}')
    print(f'q_c: {q_c.shape}')
    print(f'power_receiver: {power_receiver}')
    print(f'gamma_thr: {gamma_thr}')
    print(f'sum_power_relaxed: {np.array(sum_power_relaxed).shape}')
    print(f'sum_power_decoupled: {np.array(sum_power_decoupled).shape}')
    print(f'ap_act_sol_relaxed: {ap_act_sol_relaxed.shape}')
    print(f'ap_act_sol_decoupled: {ap_act_sol_decoupled.shape}')
    print(f'power_dl_sol_relaxed: {power_dl_sol_relaxed.shape}')
    print(f'power_dl_sol_decoupled: {power_dl_sol_decoupled.shape}')
    print(f'avg_time_relaxed: {avg_time_relaxed}')
    print(f'avg_time_decoupled: {avg_time_decoupled}')

