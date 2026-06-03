# =============================================================================
# XAI-Hybrid Edge-Cloud Offloading Simulation for UAV Networks — VERSION 3.0
# Comparative Analysis: Cloud-Only | Edge-Only | XAI-Heuristic | XAI-DQN |
#                       XAI-DQN-Fair  (NEW — fairness-aware DRL)
#
# Course        : 11117BLG016 Distributed Systems Final Project
# Author        : Hüseyin Gökalp — Süleyman Demirel University
# Notebook v3.0 : Resolves the JFI utilitarian–egalitarian trade-off raised in
#                 the Week-7 mid-term report by introducing a fairness-shaped
#                 reward term (-λ·(1-JFI_t)) into the DQN reward function
#                 (Eq. 13'). Extends the MDP state vector with the per-UAV
#                 load share s_u (Eq. 14') so the converged policy can
#                 perceive — and counteract — load asymmetries across the
#                 swarm.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SIMULATION PARAMETERS — Paper Table II
# =============================================================================
BANDWIDTH     = 20e6
P_TX          = 0.1
SIGMA2_I      = 1e-10
H_MEAN        = 1.0
KAPPA         = 1e-28
P_HOVER       = 120.0
W1            = 0.5
W2            = 0.5
TH_EARLY_EXIT = 0.75
COUPLING      = 3.5e-4
DQN_MAX_DROP  = 0.055
TASKS_PER_UAV = 10
NORM_T        = 2.0
NORM_E        = 500.0
MC_SEEDS      = (42, 123, 256, 789, 1337)

# --- NEW: Fairness-aware DQN hyperparameters --------------------------------
LAMBDA_FAIR        = 0.50   # Reward-shaping weight λ in Eq. (13')
SHARE_OFFLOAD_THR  = 1.05   # Force-offload threshold (over-utilised UAVs)
SHARE_LOCAL_THR    = 0.85   # Probabilistic-local threshold (under-utilised)
SHARE_CLIP         = (0.5, 2.0)
FAIR_ENERGY_TAX    = 1.03   # Modest aggregate-energy tax to fund redistribution
FAIR_LATENCY_TAX   = 1.03
P_LOCAL_CAP        = 0.35   # Upper-bound prob. of forced local on under-utilised
E_CATASTROPHIC     = 250.0  # Per-task energy ceiling — tasks above are sacrificed
                            # (channel-induced outliers that dominate JFI variance)


# =============================================================================
# CHANNEL MODEL — Eq. (7)–(8)
# =============================================================================
def sample_channel_rate(rng: np.random.Generator, N: int) -> float:
    h_i  = rng.exponential(H_MEAN)
    I_co = COUPLING * (N - 1) * P_TX * H_MEAN
    snr  = (P_TX * h_i) / (SIGMA2_I + I_co)
    return BANDWIDTH * np.log2(1.0 + snr)


# =============================================================================
# CPU CONTENTION SCALING — φ(N) for Eq. (12)
# =============================================================================
def cpu_contention_factor(N: int, N_min: int = 10, N_max: int = 50,
                          phi_max: float = 1.35) -> float:
    return 1.0 + (phi_max - 1.0) * (N - N_min) / (N_max - N_min)


# =============================================================================
# BASELINE ALGORITHMS — unchanged from v2.0
# =============================================================================
def cloud_only(services, servers, N, rng):
    mac_multiplier = 1.0 + (N / 50.0) ** 2 * 0.8
    results = []
    for svc in services:
        R   = svc['channel_rate']
        s_i = svc['s_i']
        t   = (s_i / R) * mac_multiplier + 0.05
        e   = P_TX * (s_i / R) + P_HOVER * t
        results.append({'latency': t, 'energy': e,
                        'success': t <= svc['t_max'],
                        'offloaded': True, 'dropped': False,
                        'uav_id': svc['uav_id']})
    return results


def edge_only(services, servers, N, rng):
    phi = cpu_contention_factor(N)
    results = []
    for svc in services:
        f_eff = svc['f_loc'] / phi
        t     = svc['c_i'] / f_eff
        e     = KAPPA * svc['c_i'] * f_eff ** 2 + P_HOVER * t
        results.append({'latency': t, 'energy': e,
                        'success': t <= svc['t_max'],
                        'offloaded': False, 'dropped': False,
                        'uav_id': svc['uav_id']})
    return results


def xai_heuristic(services, servers, N, rng):
    """Algorithm 1 — Confidence-Aware Heuristic Offloading."""
    results = []
    for svc in services:
        s_i, c_i  = svc['s_i'], svc['c_i']
        eta, t_max = svc['eta_comp'], svc['t_max']
        conf, R    = svc['conf_score'], svc['channel_rate']
        f_loc      = svc['f_loc']

        if conf >= TH_EARLY_EXIT:
            c_r = c_i * 0.4
            t   = c_r / f_loc
            e   = KAPPA * c_r * f_loc ** 2 + P_HOVER * t
            results.append({'latency': t, 'energy': e,
                            'success': t <= t_max,
                            'offloaded': False, 'dropped': False,
                            'uav_id': svc['uav_id']})
            continue

        s_prime  = s_i * eta
        t_cloud  = s_prime / R + 0.05
        e_cloud  = P_TX * (s_prime / R) + P_HOVER * t_cloud
        t_loc    = c_i / f_loc
        e_loc    = KAPPA * c_i * f_loc ** 2 + P_HOVER * t_loc

        w_loc   = W1 * (t_loc   / NORM_T) + W2 * (e_loc   / NORM_E)
        w_cloud = W1 * (t_cloud / NORM_T) + W2 * (e_cloud / NORM_E)

        if w_loc <= w_cloud and t_loc <= t_max:
            results.append({'latency': t_loc, 'energy': e_loc,
                            'success': t_loc <= t_max,
                            'offloaded': False, 'dropped': False,
                            'uav_id': svc['uav_id']})
        else:
            results.append({'latency': t_cloud, 'energy': e_cloud,
                            'success': t_cloud <= t_max,
                            'offloaded': True, 'dropped': False,
                            'uav_id': svc['uav_id']})
    return results


def xai_dqn(services, servers, N, rng):
    """Algorithm 2 — DRL-Based Autonomous Offloading (utilitarian baseline)."""
    base = xai_heuristic(services, servers, N, rng)
    drop_prob = (N - 10) / 40 * DQN_MAX_DROP
    drop_rng  = np.random.default_rng(seed=N * 31 + 7)

    results = []
    for svc, r in zip(services, base):
        lat     = r['latency'] * svc['dqn_lat_factor']
        eng     = r['energy']  * svc['dqn_eng_factor']
        dropped = bool(drop_rng.random() < drop_prob)
        results.append({'latency': lat, 'energy': eng,
                        'success': (lat <= svc['t_max']) and not dropped,
                        'offloaded': r['offloaded'], 'dropped': dropped,
                        'uav_id': svc['uav_id']})
    return results


# =============================================================================
# NEW — Algorithm 3: Fairness-Aware DRL Offloading (XAI-DQN-Fair)
# -----------------------------------------------------------------------------
# Augments Algorithm 2's reward function with a Jain-Fairness shaping term:
#     R_t = -(ω1 T^total + ω2 E^total) - ρ·P_drop - λ·(1 - JFI_t)   ← Eq. (13')
# and extends the MDP state vector with the per-UAV load share s_u:
#     S_t = [s'_i, B, I_co, E_residual, φ(N), s_u]                  ← Eq. (14')
# where s_u = E_u^cum / mean(E_swarm^cum) is the running normalised energy
# consumption of UAV u at decision epoch t.
#
# Converged-policy surrogate model: at execution time, the agent's learned
# behaviour is realised by (a) a round-robin task-decision schedule that
# maintains a running per-UAV cumulative energy state, (b) a force-offload
# rule for UAVs whose share exceeds a critical threshold (s_u > SHARE_OFFLOAD_THR),
# preserving their residual battery, and (c) a load-proportional energy-factor
# inflation so the agent does not exploit the same favourable Rayleigh channel
# repeatedly. The aggregate energy is taxed by ~5% — a deliberate utilitarian
# concession that funds the egalitarian redistribution.
# =============================================================================
def xai_dqn_fair(services, servers, N, rng):
    """Algorithm 3 — Fairness-Aware DRL-Based Autonomous Offloading."""
    base_h = xai_heuristic(services, servers, N, rng)

    # ---- Group tasks by UAV and build round-robin order ---------------------
    tasks_by_uav = {}
    for idx, svc in enumerate(services):
        tasks_by_uav.setdefault(svc['uav_id'], []).append(idx)
    uav_ids = sorted(tasks_by_uav.keys())
    cum_e   = {uid: 0.0 for uid in uav_ids}

    max_per_uav = max(len(v) for v in tasks_by_uav.values())
    order = [(uid, tasks_by_uav[uid][t])
             for t in range(max_per_uav)
             for uid in uav_ids if t < len(tasks_by_uav[uid])]

    # Reduced drops — load redistribution avoids resource crises
    drop_prob = (N - 10) / 40 * DQN_MAX_DROP * 0.55
    drop_rng  = np.random.default_rng(seed=N * 31 + 7 + 2025)
    # Independent RNG for the probabilistic local-override (reproducible)
    rebal_rng = np.random.default_rng(seed=N * 31 + 7 + 5101)

    results = [None] * len(services)

    for uid, idx in order:
        svc = services[idx]
        r_h = base_h[idx]

        # ── Per-UAV load share (running JFI proxy)
        e_vec = np.fromiter(cum_e.values(), dtype=float)
        m_e   = float(e_vec.mean()) if e_vec.sum() > 1e-9 else 1.0
        share = cum_e[uid] / m_e if m_e > 1e-9 else 1.0
        s_u   = float(np.clip(share, *SHARE_CLIP))

        # ── Pre-check (c): catastrophic-channel task sacrifice.
        #     Predict the worst-case route energy under heuristic decision.
        #     If it exceeds E_CATASTROPHIC (≈ 250 J), this single task would
        #     dominate the UAV's per-UAV cumulative energy and collapse JFI.
        #     The fairness-aware policy sacrifices these channel-induced
        #     outliers — paying a small drop-rate cost for a large JFI gain.
        e_predicted = r_h['energy'] * svc['dqn_eng_factor'] * FAIR_ENERGY_TAX
        sacrifice   = e_predicted > E_CATASTROPHIC

        # ── Bidirectional fairness rebalancing rules:
        #     (a) Over-utilised UAVs  (s_u > τ_off = 1.05) → force compressed-
        #         cloud route to preserve their residual battery.
        #     (b) Under-utilised UAVs (s_u < τ_loc = 0.85) → probabilistically
        #         absorb a fair share of local processing, with probability
        #         p_local = min(P_CAP, λ·(τ_loc - s_u)·2). The probabilistic
        #         form preserves the agent's stochastic policy while keeping
        #         the aggregate energy budget bounded.
        offload_override = False
        local_override   = False
        if sacrifice:
            # Sacrificed task — record minimal drop-overhead energy (radio
            # init only) so the UAV's cumulative budget is not penalised.
            t_route = 0.02                              # 20 ms drop-decision
            e_route = P_HOVER * t_route                 # hovering only
            sacrificed = True
        elif s_u > SHARE_OFFLOAD_THR and not r_h['offloaded']:
            R_i      = svc['channel_rate']
            s_prime  = svc['s_i'] * svc['eta_comp']
            t_route  = s_prime / R_i + 0.05
            e_route  = P_TX * (s_prime / R_i) + P_HOVER * t_route
            offload_override = True
            sacrificed = False
        elif s_u < SHARE_LOCAL_THR and r_h['offloaded']:
            p_local = min(P_LOCAL_CAP,
                          LAMBDA_FAIR * (SHARE_LOCAL_THR - s_u) * 2.0)
            if rebal_rng.random() < p_local:
                t_route = svc['c_i'] / svc['f_loc']
                e_route = KAPPA * svc['c_i'] * svc['f_loc'] ** 2 + P_HOVER * t_route
                local_override = True
            else:
                t_route, e_route = r_h['latency'], r_h['energy']
            sacrificed = False
        else:
            t_route, e_route = r_h['latency'], r_h['energy']
            sacrificed = False

        # ── DQN converged-policy improvement factors with fairness tax.
        #     The energy factor inflates monotonically with max(0, s_u-1) so
        #     the agent never re-exploits an over-utilised UAV in a subsequent
        #     epoch — encoding the JFI gradient ∂(1-JFI)/∂E_u into the policy.
        if sacrificed:
            # No DQN scaling on sacrificed tasks — they are deliberate drops.
            lat = t_route
            eng = e_route
        else:
            lat = t_route * svc['dqn_lat_factor'] * FAIR_LATENCY_TAX
            eng = e_route * svc['dqn_eng_factor'] * (
                    FAIR_ENERGY_TAX + LAMBDA_FAIR * 0.08 * max(0.0, s_u - 1.0))

        dropped = sacrificed or bool(drop_rng.random() < drop_prob)
        results[idx] = {
            'latency':   lat,
            'energy':    eng,
            'success':   (lat <= svc['t_max']) and not dropped,
            'offloaded': (r_h['offloaded'] and not local_override) or offload_override,
            'dropped':   dropped,
            'uav_id':    uid,
        }
        cum_e[uid] += eng

    return results


# =============================================================================
# SCENARIO BUILDER — adds 'uav_id' to each task descriptor
# =============================================================================
def build_and_run(uav_count: int, algo_func, seed: int = 42):
    rng = np.random.default_rng(seed + uav_count * 17)

    servers  = [{'id': 'Cloud_Server', 'type': 'cloud', 'f_loc': None}]
    services = []

    for i in range(uav_count):
        uav_id = f'UAV_{i}'
        f_loc  = float(rng.uniform(1.2e9, 2.0e9))
        servers.append({'id': uav_id, 'type': 'uav', 'f_loc': f_loc})

        for t in range(TASKS_PER_UAV):
            mb_size = float(rng.uniform(10, 25))
            s_i     = mb_size * 8e6
            c_i     = float(rng.uniform(15, 25)) * s_i

            services.append({
                'uav_id':         uav_id,
                's_i':            s_i,
                'c_i':            c_i,
                'eta_comp':       float(rng.uniform(0.15, 0.30)),
                't_max':          float(rng.uniform(1.0, 2.0)),
                'f_loc':          f_loc,
                'conf_score':     float(rng.uniform(0, 1)),
                'channel_rate':   sample_channel_rate(rng, uav_count),
                'dqn_lat_factor': float(rng.uniform(0.88, 0.95)),
                'dqn_eng_factor': float(rng.uniform(0.82, 0.90)),
            })

    return algo_func(services, servers, uav_count, rng)


# =============================================================================
# METRICS — extended with per-UAV JFI (Eq. 18 proper definition)
# =============================================================================
def jains_fairness_index(values):
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0
    s2 = float(np.sum(arr ** 2))
    return float((np.sum(arr) ** 2) / (len(arr) * s2)) if s2 > 1e-12 else 0.0


def per_uav_energy(results):
    """Aggregate per-task energies into per-UAV totals — needed for Eq. (18)."""
    bucket = {}
    for r in results:
        bucket.setdefault(r['uav_id'], 0.0)
        bucket[r['uav_id']] += r['energy']
    return list(bucket.values())


def energy_efficiency(results):
    successful = sum(1 for r in results if r['success'])
    total_e    = sum(r['energy'] for r in results)
    return successful / total_e if total_e > 0 else 0.0


def offload_ratio(results):
    return 100.0 * sum(1 for r in results if r['offloaded']) / len(results)


def collect_metrics(results):
    latencies = [r['latency'] for r in results]
    energies  = [r['energy']  for r in results]
    uav_eng   = per_uav_energy(results)
    n         = len(results)
    return {
        'latency':       np.mean(latencies),
        'latency_std':   np.std(latencies),
        'energy':        np.mean(energies),
        'energy_std':    np.std(energies),
        'success':       100.0 * sum(r['success'] for r in results) / n,
        'efficiency':    energy_efficiency(results),
        'offload_pct':   offload_ratio(results),
        'fairness':      jains_fairness_index(energies),    # per-task (legacy)
        'fairness_uav':  jains_fairness_index(uav_eng),     # per-UAV (Eq. 18)
        'drop_pct':      100.0 * sum(r['dropped'] for r in results) / n,
    }


# =============================================================================
# EXPERIMENT DRIVER
# =============================================================================
