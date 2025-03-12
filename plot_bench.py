import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# Load function
def load_data(file_name):
    with open(file_name, "rb") as f:
        data = pickle.load(f)

    return {
        "sum_power_relaxed": sum(data.get("sum_power_relaxed", [])) / len(data.get("sum_power_relaxed", [1])),
        "sum_power_decoupled": sum(data.get("sum_power_decoupled", [])) / len(data.get("sum_power_decoupled", [1]))
    }


if __name__ == '__main__':
    ues = [5, 10, 15]
    ap_40_relaxed, ap_40_decoupled = [], []
    ap_50_relaxed, ap_50_decoupled = [], []

    # Load and process data
    for ue in ues:
        for ap in [40, 50]:
            filename = f"Data/New/{ap}AP_{ue}UE_2SR.pkl"
            try:
                data = load_data(filename)
                if ap == 40:
                    ap_40_relaxed.append(data["sum_power_relaxed"])
                    ap_40_decoupled.append(data["sum_power_decoupled"])
                else:
                    ap_50_relaxed.append(data["sum_power_relaxed"])
                    ap_50_decoupled.append(data["sum_power_decoupled"])
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                if ap == 40:
                    ap_40_relaxed.append(0)
                    ap_40_decoupled.append(0)
                else:
                    ap_50_relaxed.append(0)
                    ap_50_decoupled.append(0)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(ues, ap_40_relaxed, marker='o', linestyle='-', label="40 APs - Relaxed")
    # plt.plot(ues, ap_40_decoupled, marker='s', linestyle='--', label="40 APs - Decoupled")
    plt.plot(ues, ap_50_relaxed, marker='o', linestyle='-', label="50 APs - Relaxed")
    # plt.plot(ues, ap_50_decoupled, marker='s', linestyle='--', label="50 APs - Decoupled")

    plt.xlabel("Number of UEs")
    plt.ylabel("Average Power Consumption")
    plt.title("Power Consumption vs Number of UEs for 40 and 50 APs")
    plt.xticks(ues)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show plot
    plt.show()