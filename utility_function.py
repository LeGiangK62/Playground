import numpy as np
import cvxpy as cp
import pickle
from tqdm import tqdm
import time
from itertools import combinations


def generate_data(num, num_AP, num_UE, num_SR, optimize=False):
    M = num_AP  # number of access points
    K = num_UE  # number of terminals
    T = num_SR  # number of sensing receivers
    D = 0.5  # in kilometer
    tau = 20  # training length
    U, S, V = np.linalg.svd(np.random.randn(tau, tau))
    B = 20  # MHz
    Hb = 15  # Base station height in m
    Hm = 1.65  # Mobile height in m
    f = 1900  # Frequency in MHz
    aL = (1.1 * np.log10(f) - 0.7) * Hm - (1.56 * np.log10(f) - 0.8)
    L = 46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(Hb) - aL
    power_f = 0.1  # uplink power: 100 mW
    n0 = -174
    noise_p = (B * 1e6) * 10 ** (n0 / 10)
    Pu = power_f / noise_p  # normalized receive SNR
    Pp = Pu  # pilot power

    sigma_shd = 8  # in dB
    D_cor = 0.1

    d0 = 0.01  # km
    d1 = 0.05  # km
    N = num  # realizations

    c = 3e8
    n0 = -174
    noise = (B * 1e6) * 10 ** (n0 / 10)
    zeta = (8 * (B * 1e6) ** 2 * np.pi ** 2) / (c ** 2 * noise)

    RCS_0 = 0.5  # check this
    tar_cor = np.array([0, 0])

    R_cf_min = np.zeros(N)
    R_cf_opt_min = np.zeros(N)
    R_cf_user = np.zeros((N, K))

    directs = np.zeros((N, K))
    corsses = np.zeros((N, K, K))
    betas = np.zeros((N, M, K))
    x_value = np.zeros((N, K))

    AP_cor = np.zeros((N, M, 2))
    SR_cor = np.zeros((N, T, 2))
    large_scale_fading = np.zeros((N, M, K))
    channel_variance = np.zeros((N, M, K))
    sensing_channel = np.zeros((N, M, T))

    QA = np.zeros((N, M))
    QB = np.zeros((N, M))
    QC = np.zeros((N, M))

    n = 0
    while n < N:
        valid = False
        while not valid:
            try:
                # AP
                AP = np.zeros((M, 2, 9))
                AP[:, :, 0] = np.random.uniform(-D / 2, D / 2, (M, 2))

                D1 = np.zeros((M, 2))
                D1[:, 0] = D1[:, 0] + D * np.ones(M)
                AP[:, :, 1] = AP[:, :, 0] + D1

                D2 = np.zeros((M, 2))
                D2[:, 1] = D2[:, 1] + D * np.ones(M)
                AP[:, :, 2] = AP[:, :, 0] + D2

                D3 = np.zeros((M, 2))
                D3[:, 0] = D3[:, 0] - D * np.ones(M)
                AP[:, :, 3] = AP[:, :, 0] + D3

                D4 = np.zeros((M, 2))
                D4[:, 1] = D4[:, 1] - D * np.ones(M)
                AP[:, :, 4] = AP[:, :, 0] + D4

                D5 = np.zeros((M, 2))
                D5[:, 0] = D5[:, 0] + D * np.ones(M)
                D5[:, 1] = D5[:, 1] - D * np.ones(M)
                AP[:, :, 5] = AP[:, :, 0] + D5

                D6 = np.zeros((M, 2))
                D6[:, 0] = D6[:, 0] - D * np.ones(M)
                D6[:, 1] = D6[:, 1] - D * np.ones(M)
                AP[:, :, 6] = AP[:, :, 0] + D6

                D7 = np.zeros((M, 2))
                D7 = D7 + D * np.ones((M, 2))
                AP[:, :, 7] = AP[:, :, 0] + D7

                D8 = np.zeros((M, 2))
                D8 = D8 - D * np.ones((M, 2))
                AP[:, :, 8] = AP[:, :, 0] + D8

                # SR
                SR = np.zeros((T, 2, 9))
                SR[:, :, 0] = np.random.uniform(-D / 2, D / 2, (T, 2))

                D1 = np.zeros((T, 2))
                D1[:, 0] = D1[:, 0] + D * np.ones(T)
                SR[:, :, 1] = SR[:, :, 0] + D1

                D2 = np.zeros((T, 2))
                D2[:, 1] = D2[:, 1] + D * np.ones(T)
                SR[:, :, 2] = SR[:, :, 0] + D2

                D3 = np.zeros((T, 2))
                D3[:, 0] = D3[:, 0] - D * np.ones(T)
                SR[:, :, 3] = SR[:, :, 0] + D3

                D4 = np.zeros((T, 2))
                D4[:, 1] = D4[:, 1] - D * np.ones(T)
                SR[:, :, 4] = SR[:, :, 0] + D4

                D5 = np.zeros((T, 2))
                D5[:, 0] = D5[:, 0] + D * np.ones(T)
                D5[:, 1] = D5[:, 1] - D * np.ones(T)
                SR[:, :, 5] = SR[:, :, 0] + D5

                D6 = np.zeros((T, 2))
                D6[:, 0] = D6[:, 0] - D * np.ones(T)
                D6[:, 1] = D6[:, 1] - D * np.ones(T)
                SR[:, :, 6] = SR[:, :, 0] + D6

                D7 = np.zeros((T, 2))
                D7 = D7 + D * np.ones((T, 2))
                SR[:, :, 7] = SR[:, :, 0] + D7

                D8 = np.zeros((T, 2))
                D8 = D8 - D * np.ones((T, 2))
                SR[:, :, 8] = SR[:, :, 0] + D8

                # Initialize Ter positions
                Ter = np.zeros((K, 2, 9))
                Ter[:, :, 0] = np.random.uniform(-D / 2, D / 2, (K, 2))

                D1 = np.zeros((K, 2))
                D1[:, 0] = D1[:, 0] + D * np.ones(K)
                Ter[:, :, 1] = Ter[:, :, 0] + D1

                D2 = np.zeros((K, 2))
                D2[:, 1] = D2[:, 1] + D * np.ones(K)
                Ter[:, :, 2] = Ter[:, :, 0] + D2

                D3 = np.zeros((K, 2))
                D3[:, 0] = D3[:, 0] - D * np.ones(K)
                Ter[:, :, 3] = Ter[:, :, 0] + D3

                D4 = np.zeros((K, 2))
                D4[:, 1] = D4[:, 1] - D * np.ones(K)
                Ter[:, :, 4] = Ter[:, :, 0] + D4

                D5 = np.zeros((K, 2))
                D5[:, 0] = D5[:, 0] + D * np.ones(K)
                D5[:, 1] = D5[:, 1] - D * np.ones(K)
                Ter[:, :, 5] = Ter[:, :, 0] + D5

                D6 = np.zeros((K, 2))
                D6[:, 0] = D6[:, 0] - D * np.ones(K)
                D6[:, 1] = D6[:, 1] + D * np.ones(K)
                Ter[:, :, 6] = Ter[:, :, 0] + D6

                D7 = np.zeros((K, 2))
                D7 = D7 + D * np.ones((K, 2))
                Ter[:, :, 7] = Ter[:, :, 0] + D7

                D8 = np.zeros((K, 2))
                D8 = D8 - D * np.ones((K, 2))
                Ter[:, :, 8] = Ter[:, :, 0] + D8

                Dist = np.zeros((M, M))
                Cor = np.zeros((M, M))
                for m1 in range(M):
                    for m2 in range(M):
                        Dist[m1, m2] = np.min([np.linalg.norm(AP[m1, :, 0] - AP[m2, :, i]) for i in range(9)])
                        Cor[m1, m2] = np.exp(-np.log(2) * Dist[m1, m2] / D_cor)

                A1 = np.linalg.cholesky(Cor)
                x1 = np.random.randn(M, 1)
                sh_AP = A1 @ x1
                for m in range(M):
                    sh_AP[m] = (1 / np.sqrt(2)) * sigma_shd * sh_AP[m] / np.linalg.norm(A1[m, :])

                ##
                Dist = np.zeros((T, T))
                Cor = np.zeros((T, T))
                for t1 in range(T):
                    for t2 in range(T):
                        Dist[t1, t2] = np.min([np.linalg.norm(SR[t1, :, 0] - SR[t2, :, i]) for i in range(9)])
                        Cor[t1, t2] = np.exp(-np.log(2) * Dist[t1, t2] / D_cor)

                A3 = np.linalg.cholesky(Cor)
                x3 = np.random.randn(T, 1)
                sh_SR = A3 @ x3

                for t in range(T):
                    sh_SR[t] = (1 / np.sqrt(2)) * sigma_shd * sh_SR[t] / np.linalg.norm(A3[t, :])
                ##

                Dist = np.zeros((K, K))
                Cor = np.zeros((K, K))
                for k1 in range(K):
                    for k2 in range(K):
                        Dist[k1, k2] = np.min([np.linalg.norm(Ter[k1, :, 0] - Ter[k2, :, i]) for i in range(9)])
                        Cor[k1, k2] = np.exp(-np.log(2) * Dist[k1, k2] / D_cor)

                A2 = np.linalg.cholesky(Cor)
                x2 = np.random.randn(K, 1)
                sh_Ter = A2 @ x2

                valid = True

            except np.linalg.LinAlgError:
                # print("Matrix not positive definite, retrying...")
                continue

        for k in range(K):
            sh_Ter[k] = (1 / np.sqrt(2)) * sigma_shd * sh_Ter[k] / np.linalg.norm(A2[k, :])

        Z_shd = np.zeros((M, K))
        for m in range(M):
            for k in range(K):
                Z_shd[m, k] = sh_AP[m, 0] + sh_Ter[k, 0]

        # Large-scale coefficients
        BETAA = np.zeros((M, K))
        dist = np.zeros((M, K))
        for m in range(M):
            for k in range(K):
                dist[m, k] = np.min([np.linalg.norm(AP[m, :, i] - Ter[k, :, 0]) for i in range(9)])
                index = np.argmin([np.linalg.norm(AP[m, :, i] - Ter[k, :, 0]) for i in range(9)])
                if dist[m, k] < d0:
                    betadB = -L - 35 * np.log10(d1) + 20 * np.log10(d1) - 20 * np.log10(d0)
                elif d0 <= dist[m, k] <= d1:
                    betadB = -L - 35 * np.log10(d1) + 20 * np.log10(d1) - 20 * np.log10(dist[m, k])
                else:
                    betadB = -L - 35 * np.log10(dist[m, k]) + Z_shd[m, k]
                BETAA[m, k] = 10 ** (betadB / 10)

        ## Sensing
        sens_channel = np.zeros((M, T))
        dist = np.zeros((M, T))

        Z_shd = np.zeros((M, T))
        for m in range(M):
            for t in range(T):
                Z_shd[m, t] = sh_AP[m, 0] + sh_SR[t, 0]
        for m in range(M):
            for t in range(T):
                dist[m, t] = np.min([np.linalg.norm(AP[m, :, i] - tar_cor) for i in range(9)]) + np.min(
                    [np.linalg.norm(SR[t, :, i] - tar_cor) for i in range(9)])
                if dist[m, t] < d0:
                    betadB = -L - 35 * np.log10(d1) + 20 * np.log10(d1) - 20 * np.log10(d0)
                elif d0 <= dist[m, t] <= d1:
                    betadB = -L - 35 * np.log10(d1) + 20 * np.log10(d1) - 20 * np.log10(dist[m, t])
                else:
                    betadB = -L - 35 * np.log10(dist[m, t]) + Z_shd[m, t]
                sens_channel[m, t] = 10 ** (betadB / 10) * RCS_0

        # Pilot assignment: (random choice)
        Phii = np.zeros((tau, K))
        for k in range(K):
            Point = k
            Phii[:, k] = U[:, Point]

        Phii_cf = Phii

        # Compute Gamma matrix
        Gammaa = np.zeros((M, K))
        mau = np.zeros((M, K))
        for m in range(M):
            for k in range(K):
                mau[m, k] = np.linalg.norm((BETAA[m, :] ** (1 / 2) * (Phii_cf[:, k].T @ Phii_cf))) ** 2
                Gammaa[m, k] = tau * Pp * BETAA[m, k] ** 2 / (tau * Pp * mau[m, k] + 1)

        # SINR and rate calculation
        SINR = np.zeros(K)
        R_cf = np.zeros(K)

        PC = np.zeros((K, K))
        for ii in range(K):
            for k in range(K):
                PC[ii, k] = np.sum((Gammaa[:, k] / BETAA[:, k] * BETAA[:, ii]) * (Phii_cf[:, k].T @ Phii_cf[:, ii]))
        PC1 = (np.abs(PC)) ** 2

        for k in range(K):
            deno1 = 0
            for m in range(M):
                deno1 += Gammaa[m, k] * np.sum(BETAA[m, :])

            SINR[k] = Pu * (np.sum(Gammaa[:, k])) ** 2 / (
                        np.sum(Gammaa[:, k]) + Pu * deno1 + Pu * np.sum(PC1[:, k]) - Pu * PC1[k, k])
            R_cf[k] = np.log2(1 + SINR[k])

        stepp = 5
        Ratestep = np.zeros((stepp, K))
        Ratestep[0, :] = R_cf

        for st in range(1, stepp):
            minvalue, minindex = np.min(Ratestep[st - 1, :]), np.argmin(Ratestep[st - 1, :])
            Mat = np.zeros((tau, tau)) - Pu * np.sum(BETAA[:, minindex]) * np.outer(Phii_cf[:, minindex],
                                                                                    Phii_cf[:, minindex])
            for kk in range(K):
                Mat += Pu * np.sum(BETAA[:, kk]) * (Phii_cf[:, kk] @ Phii_cf[:, kk].T)

            U1, S1, V1 = np.linalg.svd(Mat, full_matrices=True)
            Phii_cf[:, minindex] = U1[:, tau - 1]

        Gammaa = np.zeros((M, K))
        mau = np.zeros((M, K))

        for m in range(M):
            for k in range(K):
                mau[m, k] = np.linalg.norm(
                    (BETAA[m, :] ** 0.5) * (Phii_cf[:, k].T @ Phii_cf) ** 2
                ) ** 2
        for m in range(M):
            for k in range(K):
                Gammaa[m, k] = tau * Pp * BETAA[m, k] ** 2 / (tau * Pp * mau[m, k] + 1)

        SINR = np.zeros(K)
        PC = np.zeros((K, K))

        for ii in range(K):
            for k in range(K):
                PC[ii, k] = np.sum((Gammaa[:, k] / BETAA[:, k]) * BETAA[:, ii]) * np.dot(Phii_cf[:, k], Phii_cf[:, ii])
        PC1 = np.abs(PC) ** 2
        for k in range(K):
            deno1 = 0
            for m in range(M):
                deno1 += Gammaa[m, k] * np.sum(BETAA[m, :])
            SINR[k] = Pu * (np.sum(Gammaa[:, k])) ** 2 / (
                    np.sum(Gammaa[:, k]) + Pu * deno1 + Pu * np.sum(PC1[:, k]) - Pu * PC1[k, k]
            )
            # Rate calculation
            Ratestep[st - 1, k] = np.log2(1 + SINR[k])

        R_cf_min[n] = np.min(Ratestep[stepp - 1, :])
        R_cf_user[n, :] = Ratestep[stepp - 1, :]

        tmin = 2 ** R_cf_min[n] - 1
        tmax = 2 ** (2 * R_cf_min[n] + 1.2) - 1
        epsi = max(tmin / 5, 0.01)

        BETAAn = BETAA * Pu
        Gammaan = Gammaa * Pu
        PhiPhi = np.zeros((K, K))
        Te1 = np.zeros((K, K))
        Te2 = np.zeros((K, K))
        direct = np.zeros(K)

        for ii in range(K):
            for k in range(K):
                PhiPhi[ii, k] = np.linalg.norm(Phii_cf[:, ii].T @ Phii_cf[:, k])

        for ii in range(K):
            direct[ii] = np.sum(Gammaan[:, ii])
            for k in range(K):
                Te1[ii, k] = np.sum(BETAAn[:, ii] * Gammaan[:, k])
                Te2[ii, k] = (
                        np.sum((Gammaan[:, k] / BETAA[:, k]) * BETAA[:, ii]) ** 2
                        * PhiPhi[k, ii] ** 2
                )
                if ii == k:
                    Te2[ii, k] = 0

        cross = Te1 + Te2
        directs[n, :] = direct
        corsses[n, :, :] = cross
        betas[n, :, :] = BETAAn

        for m in range(M):
            R_AP = np.min([np.linalg.norm(AP[m, :, i] - tar_cor) for i in range(9)])
            for t in range(T):
                R_SR = np.min([np.linalg.norm(SR[t, :, i] - tar_cor) for i in range(9)])
                QA[n, m] += sens_channel[m, t] * (
                            (AP[m, 0, 0] - tar_cor[0]) / R_AP + (SR[t, 0, 0] - tar_cor[0]) / R_SR) ** 2
                QB[n, m] += sens_channel[m, t] * (
                            (AP[m, 1, 0] - tar_cor[1]) / R_AP + (SR[t, 1, 0] - tar_cor[1]) / R_SR) ** 2
                QC[n, m] += sens_channel[m, t] * (
                            (AP[m, 0, 0] - tar_cor[0]) / R_AP + (SR[t, 0, 0] - tar_cor[0]) / R_SR) * (
                                        (AP[m, 1, 0] - tar_cor[1]) / R_AP + (SR[t, 1, 0] - tar_cor[1]) / R_SR)
            QA[n, m] = QA[n, m] * zeta
            QB[n, m] = QB[n, m] * zeta
            QC[n, m] = QC[n, m] * zeta

        # A = np.matmul(q_a,q_b.T) - np.matmul(q_c,q_c.T)
        # eigenvalues = np.linalg.eigvalsh(A)  # Use eigvalsh for symmetric/hermitian matrices

        # if np.all(eigenvalues >= 0):
        #     valid = True
        AP_cor[n, :, :] = (AP[:, :, 0] + D / 2) / D
        SR_cor[n, :, :] = (SR[:, :, 0] + D / 2) / D
        large_scale_fading[n, :, :] = BETAA / noise_p
        channel_variance[n, :, :] = Gammaa / noise_p
        sensing_channel[n, :, :] = sens_channel
        n += 1

    tar_cor = tar_cor + D / 2
    large_scale_fading = np.log1p(large_scale_fading)
    channel_variance = np.log1p(channel_variance)
    return large_scale_fading, channel_variance, sensing_channel, AP_cor, SR_cor, np.tile(tar_cor,
                                                                                          (N, 1, 1)), Pu, QA, QB, QC


def all_bench(num_sample, num_APs, num_UEs, num_SRs, gamma_thr, nu,
              power_receiver, p_dl, data_file_name):
    varsigma_all = np.zeros((num_sample, num_APs, num_UEs))
    v_all = np.zeros((num_sample, num_APs, num_UEs))
    RCS_all = np.zeros((num_sample, num_APs, num_SRs))
    AP_loc_all = np.zeros((num_sample, num_APs, 2))
    SR_loc_all = np.zeros((num_sample, num_SRs, 2))
    tar_loc_all = np.zeros((num_sample, 1, 2))
    Pd_all = np.zeros((num_sample, num_APs, num_UEs))
    q_a_all = np.zeros((num_sample, num_APs))
    q_b_all = np.zeros((num_sample, num_APs))
    q_c_all = np.zeros((num_sample, num_APs))

    power_sol_all_relaxed = np.zeros((num_sample, num_APs, num_UEs))
    power_sol_all_relaxed_full = np.zeros((num_sample, num_APs, num_UEs))
    power_sol_all_decoupled = np.zeros((num_sample, num_APs, num_UEs))
    sum_power_all_relaxed = []
    sum_power_all_relaxed_full = []
    sum_power_all_decoupled = []

    total_time_relaxed = 0
    total_time_relaxed_full = 0
    total_time_decoupled = 0
    count_relaxed = 0
    count_relaxed_full = 0
    count_decoupled = 0

    eachSample = 0
    count = 0

    with tqdm(total=num_sample, desc="Generating Samples", unit="sample") as pbar:
        while eachSample < num_sample:
            count += 1

            varsigma, v, RCS, AP_loc, SR_loc, tar_loc, Pd, q_a, q_b, q_c = generate_data(
                num=1, num_AP=num_APs, num_UE=num_UEs, num_SR=num_SRs, optimize=False
            )

            # Relaxed Constraints
            start_time = time.time()
            power_dl_sol = relaxed_optimization(v[0], varsigma[0], q_a[0], q_b[0], q_c[0], nu, gamma_thr, p_dl)
            if power_dl_sol is not None:
                count_relaxed_full += 1
                total_time_relaxed_full += time.time() - start_time
                power_dl_relaxed_sol_full = power_dl_sol  # Full

                start_time = time.time()
                random_indices = np.random.permutation(num_APs)
                for i in range(1, num_APs + 1):
                    rru_activation = np.zeros(num_APs, dtype=int)
                    rru_activation[random_indices[:i]] = 1

                    varsigma_tmp = varsigma[0][random_indices[:i], :]
                    v_tmp = v[0][random_indices[:i], :]
                    q_a_tmp = q_a[0][random_indices[:i]]
                    q_b_tmp = q_b[0][random_indices[:i]]
                    q_c_tmp = q_c[0][random_indices[:i]]
                    power_dl_sol = relaxed_optimization(v_tmp, varsigma_tmp, q_a_tmp, q_b_tmp, q_c_tmp, nu, gamma_thr, p_dl)

                    if power_dl_sol is not None:
                        power_dl_full = np.zeros((num_APs, num_UEs))
                        power_dl_full[random_indices[:i], :] = power_dl_sol
                        break

                total_time_relaxed += time.time() - start_time
                count_relaxed += 1
                power_dl_relaxed_sol = power_dl_full
                ap_act_relaxed_sol = rru_activation
            else:
                continue

            # Decoupled Constraints
            start_time = time.time()
            random_indices = np.random.permutation(num_APs)
            for i in range(1, num_APs + 1):
                rru_activation = np.zeros(num_APs, dtype=int)
                rru_activation[random_indices[:i]] = 1

                varsigma_tmp = varsigma[0][random_indices[:i], :]
                v_tmp = v[0][random_indices[:i], :]
                q_a_tmp = q_a[0][random_indices[:i]]
                q_b_tmp = q_b[0][random_indices[:i]]
                q_c_tmp = q_c[0][random_indices[:i]]
                power_dl_sol = decouple_optimization(v_tmp, varsigma_tmp, q_a_tmp, q_b_tmp, q_c_tmp, nu, gamma_thr, p_dl)

                if power_dl_sol is not None:
                    power_dl_full = np.zeros((num_APs, num_UEs))
                    power_dl_full[random_indices[:i], :] = power_dl_sol
                    break

            total_time_decoupled += time.time() - start_time
            count_decoupled += 1
            power_dl_decoupled_sol = power_dl_full
            ap_act_decoupled_sol = rru_activation

            if power_dl_relaxed_sol_full is None or power_dl_relaxed_sol is None or power_dl_decoupled_sol is None:
                continue

            sum_power_relaxed = np.mean(np.sum(power_dl_relaxed_sol, axis=(0, 1)) + power_receiver * np.sum(ap_act_relaxed_sol))
            sum_power_relaxed_full = np.mean(np.sum(power_dl_relaxed_sol_full, axis=(0, 1)) + power_receiver * num_APs)
            sum_power_decoupled = np.mean(np.sum(power_dl_decoupled_sol, axis=(0, 1)) + power_receiver * np.sum(ap_act_decoupled_sol))

            varsigma_all[eachSample, :, :] = varsigma[0]
            v_all[eachSample, :, :] = v[0]
            RCS_all[eachSample, :, :] = RCS[0]
            AP_loc_all[eachSample, :, :] = AP_loc[0]
            SR_loc_all[eachSample, :, :] = SR_loc[0]
            tar_loc_all[eachSample, :, :] = tar_loc[0]
            q_a_all[eachSample, :] = q_a[0]
            q_b_all[eachSample, :] = q_b[0]
            q_c_all[eachSample, :] = q_c[0]

            power_sol_all_relaxed[eachSample, :, :] = power_dl_relaxed_sol
            power_sol_all_relaxed_full[eachSample, :, :] = power_dl_relaxed_sol
            power_sol_all_decoupled[eachSample, :, :] = power_dl_decoupled_sol
            sum_power_all_relaxed.append(sum_power_relaxed)
            sum_power_all_relaxed_full.append(sum_power_relaxed_full)
            sum_power_all_decoupled.append(sum_power_decoupled)

            eachSample += 1
            pbar.update(1)

    avg_time_relaxed = total_time_relaxed / num_sample
    avg_time_relaxed_full = total_time_relaxed_full / num_sample
    avg_time_decoupled = total_time_decoupled / num_sample
    data_dict = {
        "varsigma": varsigma_all,
        "v": v_all,
        "RCS": RCS_all,
        "AP_loc": AP_loc_all,
        "SR_loc": SR_loc_all,
        "tar_loc": tar_loc_all,
        "Pd": p_dl,
        "q_a": q_a_all,
        "q_b": q_b_all,
        "q_c": q_c_all,
        "power_dl_sol_relaxed": power_sol_all_relaxed,
        "power_dl_sol_relaxed_full": power_sol_all_relaxed_full,
        "power_dl_sol_decoupled": power_sol_all_decoupled,
        "sum_power_relaxed": sum_power_all_relaxed,
        "sum_power_relaxed_full": sum_power_all_relaxed_full,
        "sum_power_decoupled": sum_power_all_decoupled,
        "avg_time_relaxed": avg_time_relaxed,
        "avg_time_relaxed_full": avg_time_relaxed_full,
        "avg_time_decoupled": avg_time_decoupled,
        "num_APs": num_APs,
        "num_UEs": num_UEs,
        "num_SRs": num_SRs,
        "gamma_thr": gamma_thr,
        "nu": nu,
        "power_receiver": power_receiver
    }

    with open(data_file_name, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"All results saved to {data_file_name}")
    print(f"Average execution time (Relaxed): {avg_time_relaxed:.6f} seconds")
    print(f"Average execution time (Decoupled): {avg_time_decoupled:.6f} seconds")


def relaxed_optimization(v, varsigma, q_a, q_b, q_c, nu, gamma_thr, p_dl, verbose=False):
    # Carefully check the normalization of input
    M, K = varsigma.shape
    q_a = q_a[:, None]
    q_b = q_b[:, None]
    q_c = q_c[:, None]
    b = q_a + q_b  # Constant matrix
    A = np.matmul(q_a, q_b.T) - np.matmul(q_c, q_c.T)  # Constant matrix

    y = cp.Variable((M, K), nonneg=True)  # y_{mk} = sqrt(rho_{mk})
    t = cp.Variable(K, nonneg=True)

    constraints = []
    for k in range(K):
        MI_k = cp.sum(cp.multiply(cp.square(y[:, k]), v[:, k] * varsigma[:, k]))

        BU_k = 0
        for k_prime in range(K):
            if k_prime != k:
                BU_k += cp.sum(cp.square(cp.multiply(y[:, k_prime], v[:, k] * varsigma[:, k] / varsigma[:, k_prime])))
        constraints.append(BU_k + MI_k + 1 <= cp.sqrt(t[k]))
        rhs = 0
        for m in range(M):
            rhs += cp.multiply(y[m, k], v[m, k])
        constraints.append(cp.SOC(rhs, cp.vstack([cp.multiply(np.sqrt(gamma_thr), t[k])])))

    # rho = cp.square(y)
    constraints.append(cp.sum(cp.square(y), axis=1) <= p_dl)
    rho = cp.Variable((M, K), nonneg=True)
    constraints.append(rho >= cp.square(y))

    rho_sens = cp.sum(rho, axis=1, keepdims=False)
    Ap = A @ rho_sens
    crlb = b - nu * Ap

    constraints.append(crlb <= 0)

    objective = cp.Minimize(cp.sum(cp.square(y)))

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.MOSEK)
    except cp.error.SolverError:
        pass
    if problem.status == "optimal":
        optimized_y_values = np.square(y.value)  # Compute power allocations
    else:
        optimized_y_values = None  # Indicate failure

    return optimized_y_values


def relaxed_optimization_wrong(v, varsigma, q_a, q_b, q_c, nu, gamma_thr, verbose=False):
    # Carefully check the normalization of input
    M, K = varsigma.shape
    q_a = q_a[:, None]
    q_b = q_b[:, None]
    q_c = q_c[:, None]
    b = q_a + q_b  # Constant matrix
    A = np.matmul(q_a, q_b.T) - np.matmul(q_c, q_c.T)  # Constant matrix

    y = cp.Variable((M, K), nonneg=True)  # y_{mk} = sqrt(rho_{mk})
    rho = cp.Variable((M, K), nonneg=True)

    constraints = []
    for k in range(K):  # for each UE SINR constraint
        lhs = 1
        BU = 0
        for k_prime in range(K):
            for m in range(M):
                lhs += cp.multiply(cp.square(y[m, k_prime]), v[m, k_prime] * varsigma[m, k])
            if k_prime != k:
                for m in range(M):
                    term = cp.multiply(y[m, k_prime], v[m, k] * varsigma[m, k] / varsigma[m, k_prime])
                BU += cp.square(term)
        lhs += BU
        lhs = cp.multiply(np.sqrt(gamma_thr), cp.square(lhs))
        ##
        rhs = 0
        for m in range(M):
            rhs += cp.multiply(y[m, k], v[m, k])
        constraints.append(lhs <= rhs)

    # rho = cp.square(y)
    constraints.append(rho >= cp.square(y))
    # constraints.append(cp.sum(cp.square(y), axis=1) <= p_dl)
    rho_sens = cp.sum(rho, axis=1, keepdims=True)
    Ap = A @ rho_sens
    crlb = b - nu * Ap

    constraints.append(crlb <= 0)

    objective = cp.Minimize(cp.sum(cp.square(y)))
    # objective = cp.Minimize(cp.sum(rho))

    problem = cp.Problem(objective, constraints)
    # problem.solve()
    problem.solve(solver=cp.MOSEK, verbose=True)

    return np.square(y.value)


def decouple_optimization(v, varsigma, q_a, q_b, q_c, nu, gamma_thr, p_dl, verbose=False):
    # Carefully check the normalization of input
    M, K = varsigma.shape
    q_a = q_a[:, None]
    q_b = q_b[:, None]
    q_c = q_c[:, None]
    b = q_a + q_b  # Constant matrix
    A = np.matmul(q_a, q_b.T) - np.matmul(q_c, q_c.T)  # Constant matrix

    y = cp.Variable((M, K), nonneg=True)  # y_{mk} = sqrt(rho_{mk})
    t = cp.Variable(K, nonneg=True)

    constraints = []
    for k in range(K):
        MI_k = cp.sum(cp.multiply(cp.square(y[:, k]), v[:, k] * varsigma[:, k]))

        BU_k = 0
        for k_prime in range(K):
            if k_prime != k:
                BU_k += cp.sum(cp.square(cp.multiply(y[:, k_prime], v[:, k] * varsigma[:, k] / varsigma[:, k_prime])))
        constraints.append(BU_k + MI_k + 1 <= cp.sqrt(t[k]))
        rhs = 0
        for m in range(M):
            rhs += cp.multiply(y[m, k], v[m, k])
        constraints.append(cp.SOC(rhs, cp.vstack([cp.multiply(np.sqrt(gamma_thr), t[k])])))

    objective = cp.Minimize(cp.sum(cp.square(y)))
    constraints.append(cp.sum(cp.square(y), axis=1) <= p_dl)
    # objective = cp.Minimize(cp.sum(rho))

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.MOSEK)
    except cp.error.SolverError:
        pass
    if problem.status == "optimal":
        opt_power = np.square(y.value)  # Compute power allocations
        opt_power_AP = np.sum(opt_power, axis=1, keepdims=True)
        num = b.T @ opt_power_AP
        denom = nu * opt_power_AP.T @ A @ opt_power_AP
        scale = num / denom
        scale = max(scale[0, 0], 1)
        opt_power = np.array(opt_power) * scale
        opt_power = np.minimum(opt_power, p_dl)
    else:
        opt_power = None  # Indicate failure

    return opt_power


def generate_subsets(init_set):
    """Generate all subsets of size len(init_set) - 1 from a given initial set."""
    if len(init_set) <= 1:
        return []  # Stop if we only have one AP left
    subsets = list(combinations(init_set, len(init_set) - 1))
    return [set(subset) for subset in subsets]


def bb_relaxed(num_APs, num_UEs, v, varsigma, q_a, q_b, q_c, nu, gamma_thr, p_dl):
    """
    Branch-and-Bound with Constraint Relaxation for Active AP Selection and Power Allocation.
    """
    cur_level = num_APs
    best_solution = None
    best_AP_set = set(range(num_APs))
    while cur_level > 1:  # Stop when we have at least one AP
        all_sets = generate_subsets(best_AP_set)
        best_solution_perlvl = None
        best_AP_set_perlvl = None
        best_sum_perlvl = float('inf')

        for current_set in all_sets:

            # Extract subset of parameters
            varsigma_tmp = varsigma[0][list(current_set), :]
            v_tmp = v[0][list(current_set), :]
            q_a_tmp = q_a[0][list(current_set)]
            q_b_tmp = q_b[0][list(current_set)]
            q_c_tmp = q_c[0][list(current_set)]

            # Solve power allocation
            power_dl_sol = relaxed_optimization(v_tmp, varsigma_tmp, q_a_tmp, q_b_tmp, q_c_tmp, nu, gamma_thr, p_dl)

            if power_dl_sol is not None:
                power_dl_full = np.zeros((num_APs, num_UEs))
                power_dl_full[list(current_set), :] = power_dl_sol
                sum_power = np.sum(power_dl_sol)

                if best_solution_perlvl is None or sum_power < best_sum_perlvl:
                    best_solution_perlvl = power_dl_full
                    best_sum_perlvl = sum_power
                    best_AP_set_perlvl = current_set

        # Update the best found solution at this level
        if best_sum_perlvl != float('inf'):
            best_solution = best_solution_perlvl
            best_AP_set = best_AP_set_perlvl
            cur_level -= 1
        else:
            break  # Stop if no valid solutions were found
    ap_act_vector = np.zeros(num_APs, dtype=int)  # Initialize a zero vector
    ap_act_vector[list(best_AP_set)] = 1

    return ap_act_vector, best_solution


def bb_decouple(num_APs, num_UEs, v, varsigma, q_a, q_b, q_c, nu, gamma_thr, p_dl):
    """
    Branch-and-Bound with Constraint Relaxation for Active AP Selection and Power Allocation.
    """
    cur_level = num_APs
    best_solution = None
    best_AP_set = set(range(num_APs))
    while cur_level > 1:  # Stop when we have at least one AP
        all_sets = generate_subsets(best_AP_set)
        best_solution_perlvl = None
        best_AP_set_perlvl = None
        best_sum_perlvl = float('inf')

        for current_set in all_sets:
            # Extract subset of parameters
            varsigma_tmp = varsigma[0][list(current_set), :]
            v_tmp = v[0][list(current_set), :]
            q_a_tmp = q_a[0][list(current_set)]
            q_b_tmp = q_b[0][list(current_set)]
            q_c_tmp = q_c[0][list(current_set)]

            # Solve power allocation
            power_dl_sol = decouple_optimization(v_tmp, varsigma_tmp, q_a_tmp, q_b_tmp, q_c_tmp, nu, gamma_thr, p_dl)

            if power_dl_sol is not None:
                power_dl_full = np.zeros((num_APs, num_UEs))
                power_dl_full[list(current_set), :] = power_dl_sol
                sum_power = np.sum(power_dl_sol)

                if best_solution_perlvl is None or sum_power < best_sum_perlvl:
                    best_solution_perlvl = power_dl_full
                    best_sum_perlvl = sum_power
                    best_AP_set_perlvl = current_set

        # Update the best found solution at this level
        if best_sum_perlvl != float('inf'):
            best_solution = best_solution_perlvl
            best_AP_set = best_AP_set_perlvl
            cur_level -= 1
        else:
            break  # Stop if no valid solutions were found

    ap_act_vector = np.zeros(num_APs, dtype=int)  # Initialize a zero vector
    ap_act_vector[list(best_AP_set)] = 1

    return ap_act_vector, best_solution


def generate_all_data(num_sample, num_APs, num_UEs, num_SRs, gamma_thr, nu,
              power_receiver, p_dl, data_file_name):
    varsigma_all = np.zeros((num_sample, num_APs, num_UEs))
    v_all = np.zeros((num_sample, num_APs, num_UEs))
    RCS_all = np.zeros((num_sample, num_APs, num_SRs))
    AP_loc_all = np.zeros((num_sample, num_APs, 2))
    SR_loc_all = np.zeros((num_sample, num_SRs, 2))
    tar_loc_all = np.zeros((num_sample, 1, 2))
    Pd_all = np.zeros((num_sample, num_APs, num_UEs))
    q_a_all = np.zeros((num_sample, num_APs))
    q_b_all = np.zeros((num_sample, num_APs))
    q_c_all = np.zeros((num_sample, num_APs))

    power_sol_all_relaxed = np.zeros((num_sample, num_APs, num_UEs))
    power_sol_all_relaxed_full = np.zeros((num_sample, num_APs, num_UEs))
    power_sol_all_decoupled = np.zeros((num_sample, num_APs, num_UEs))

    ap_act_all_relaxed = np.zeros((num_sample, num_APs), dtype=int)
    ap_act_all_decoupled = np.zeros((num_sample, num_APs), dtype=int)

    sum_power_all_relaxed = []
    sum_power_all_relaxed_full = []
    sum_power_all_decoupled = []

    total_time_relaxed = 0
    total_time_relaxed_full = 0
    total_time_decoupled = 0
    count_relaxed = 0
    count_relaxed_full = 0
    count_decoupled = 0

    eachSample = 0
    count = 0

    with tqdm(total=num_sample, desc="Generating Samples", unit="sample") as pbar:
        while eachSample < num_sample:
            count += 1

            varsigma, v, RCS, AP_loc, SR_loc, tar_loc, Pd, q_a, q_b, q_c = generate_data(
                num=1, num_AP=num_APs, num_UE=num_UEs, num_SR=num_SRs, optimize=False
            )

            # Relaxed Constraints
            start_time = time.time()
            power_dl_sol = relaxed_optimization(v[0], varsigma[0], q_a[0], q_b[0], q_c[0], nu, gamma_thr, p_dl)
            if power_dl_sol is not None:
                count_relaxed_full += 1
                total_time_relaxed_full += time.time() - start_time
                power_dl_relaxed_sol_full = power_dl_sol  # Full

                start_time = time.time()
                ap_act_relaxed_sol, power_dl_relaxed_sol = bb_relaxed(num_APs, num_UEs, v, varsigma, q_a, q_b, q_c, nu,
                                                          gamma_thr, p_dl)

                total_time_relaxed += time.time() - start_time
                count_relaxed += 1
            else:
                continue

            # Decoupled Constraints
            start_time = time.time()
            ap_act_decoupled_sol, power_dl_decoupled_sol = bb_decouple(num_APs, num_UEs, v, varsigma, q_a, q_b, q_c, nu,
                                                       gamma_thr, p_dl)

            total_time_decoupled += time.time() - start_time
            count_decoupled += 1

            if power_dl_relaxed_sol_full is None or power_dl_relaxed_sol is None or power_dl_decoupled_sol is None:
                continue

            sum_power_relaxed = np.mean(np.sum(power_dl_relaxed_sol, axis=(0, 1)) + power_receiver * np.sum(ap_act_relaxed_sol))
            sum_power_relaxed_full = np.mean(np.sum(power_dl_relaxed_sol_full, axis=(0, 1)) + power_receiver * num_APs)
            sum_power_decoupled = np.mean(np.sum(power_dl_decoupled_sol, axis=(0, 1)) + power_receiver * np.sum(ap_act_decoupled_sol))

            varsigma_all[eachSample, :, :] = varsigma[0]
            v_all[eachSample, :, :] = v[0]
            RCS_all[eachSample, :, :] = RCS[0]
            AP_loc_all[eachSample, :, :] = AP_loc[0]
            SR_loc_all[eachSample, :, :] = SR_loc[0]
            tar_loc_all[eachSample, :, :] = tar_loc[0]
            q_a_all[eachSample, :] = q_a[0]
            q_b_all[eachSample, :] = q_b[0]
            q_c_all[eachSample, :] = q_c[0]

            power_sol_all_relaxed[eachSample, :, :] = power_dl_relaxed_sol
            power_sol_all_relaxed_full[eachSample, :, :] = power_dl_relaxed_sol
            power_sol_all_decoupled[eachSample, :, :] = power_dl_decoupled_sol

            ap_act_all_relaxed[eachSample, :] = ap_act_relaxed_sol
            ap_act_all_decoupled[eachSample, :] = ap_act_decoupled_sol

            sum_power_all_relaxed.append(sum_power_relaxed)
            sum_power_all_relaxed_full.append(sum_power_relaxed_full)
            sum_power_all_decoupled.append(sum_power_decoupled)

            eachSample += 1
            pbar.update(1)

    avg_time_relaxed = total_time_relaxed / num_sample
    avg_time_relaxed_full = total_time_relaxed_full / num_sample
    avg_time_decoupled = total_time_decoupled / num_sample
    data_dict = {
        "varsigma": varsigma_all,
        "v": v_all,
        "RCS": RCS_all,
        "AP_loc": AP_loc_all,
        "SR_loc": SR_loc_all,
        "tar_loc": tar_loc_all,
        "Pd": p_dl,
        "q_a": q_a_all,
        "q_b": q_b_all,
        "q_c": q_c_all,
        "power_dl_sol_relaxed": power_sol_all_relaxed,
        "power_dl_sol_relaxed_full": power_sol_all_relaxed_full,
        "power_dl_sol_decoupled": power_sol_all_decoupled,
        "ap_act_relaxed_sol": ap_act_all_relaxed,
        "ap_act_decoupled_sol": ap_act_all_decoupled,
        "sum_power_relaxed": sum_power_all_relaxed,
        "sum_power_relaxed_full": sum_power_all_relaxed_full,
        "sum_power_decoupled": sum_power_all_decoupled,
        "avg_time_relaxed": avg_time_relaxed,
        "avg_time_relaxed_full": avg_time_relaxed_full,
        "avg_time_decoupled": avg_time_decoupled,
        "num_APs": num_APs,
        "num_UEs": num_UEs,
        "num_SRs": num_SRs,
        "gamma_thr": gamma_thr,
        "nu": nu,
        "power_receiver": power_receiver
    }

    with open(data_file_name, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"All results saved to {data_file_name}")
    print(f"Average execution time (Relaxed): {avg_time_relaxed:.6f} seconds")
    print(f"Average execution time (Decoupled): {avg_time_decoupled:.6f} seconds")

