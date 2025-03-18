import numpy as np
import cvxpy as cp
from utility_function import generate_data


def relaxed_optimization(beta, gamma, gamma_thr, p_dl, verbose=False):
    # Carefully check the normalization of input
    M, K = beta.shape

    y = cp.Variable((M, K), nonneg=True)  # y_{mk} = sqrt(rho_{mk})
    t = cp.Variable(K, nonneg=True)

    constraints = []
    for k in range(K):
        MI_k = cp.sum(cp.multiply(cp.square(y[:, k]), gamma[:, k] * beta[:, k]))

        BU_k = 0
        for k_prime in range(K):
            if k_prime != k:
                BU_k += cp.sum(cp.square(cp.multiply(y[:, k_prime], gamma[:, k] * beta[:, k] / beta[:, k_prime])))
        constraints.append(BU_k + MI_k + 1 <= cp.sqrt(t[k]))
        rhs = 0
        for m in range(M):
            rhs += cp.multiply(y[m, k], gamma[m, k])
        constraints.append(cp.SOC(rhs, cp.vstack([cp.multiply(np.sqrt(gamma_thr), t[k])])))

    # rho = cp.square(y)
    constraints.append(cp.sum(cp.square(y), axis=1) <= p_dl)

    objective = cp.Minimize(cp.sum(cp.square(y)))

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.MOSEK)
    except cp.error.SolverError:
        pass
    if problem.status == "optimal":
        optimized_y_values = np.square(y.value)
    else:
        optimized_y_values = None

    return optimized_y_values


def find_max_UEs(beta, gamma, gamma_thr, p_dl, num_UEs):
    random_indices = np.random.permutation(num_UEs)
    idle_UEs = list(random_indices)
    cur_UE = []
    max_supported_UEs = 0
    best_power = None

    while idle_UEs:
        cur_UE.append(idle_UEs.pop(0))
        beta_tmp = beta[0][:, cur_UE]
        gamma_tmp = gamma[0][:, cur_UE]

        power_dl_sol = relaxed_optimization(beta_tmp, gamma_tmp, gamma_thr, p_dl)

        if power_dl_sol is not None:
            max_supported_UEs = len(cur_UE)
            best_power = power_dl_sol
        else:
            break

    return max_supported_UEs, best_power


