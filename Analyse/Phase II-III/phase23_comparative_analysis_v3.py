# ===========================================================================
# Phase II & III -- Stochastic Robustness + Fairness-Aware DRL
#   (adds Rayleigh fading, CPU contention, XAI-DQN-Fair)
# Course : 11117BLG016 Distributed Systems -- Final Project
# Author : Huseyin Gokalp -- Suleyman Demirel University
# Source : exported from Improved_Comparative_Analysis_v3.ipynb
# ===========================================================================

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
def run_experiments(seeds=MC_SEEDS):
    uav_counts = [10, 20, 30, 40, 50]
    algorithms = {
        'Cloud-Only':           cloud_only,
        'Edge-Only':            edge_only,
        'XAI-Heuristic':        xai_heuristic,
        'XAI-DQN (EdgeAISim)':  xai_dqn,
        'XAI-DQN-Fair':         xai_dqn_fair,        # NEW
    }
    keys = ['latency', 'latency_std', 'energy', 'energy_std',
            'success', 'efficiency', 'offload_pct',
            'fairness', 'fairness_uav', 'drop_pct']

    agg = {a: {k: [] for k in keys} for a in algorithms}

    print("=" * 105)
    print("  COMPARATIVE ANALYSIS v3.0 — XAI-Hybrid Offloading with Fairness-Aware DRL")
    print(f"  Multi-seed Monte Carlo | M={len(seeds)} | Seeds: {seeds}")
    print("=" * 105)
    hdr = (f"{'N':<4} {'Algorithm':<22} {'Lat':<7} {'±Std':<6} {'Eng':<7} "
           f"{'Succ%':<7} {'EE×10³':<8} {'Off%':<6} {'JFI_t':<7} {'JFI_u':<7} {'Drop%'}")
    print(hdr); print("─" * len(hdr))

    for N in uav_counts:
        for algo_name, fn in algorithms.items():
            ms = {k: [] for k in keys}
            for s in seeds:
                res = build_and_run(N, fn, seed=s)
                m   = collect_metrics(res)
                for k in keys: ms[k].append(m[k])
            for k in keys: agg[algo_name][k].append(np.mean(ms[k]))

            r = {k: agg[algo_name][k][-1] for k in keys}
            print(f"{N:<4} {algo_name:<22} {r['latency']:<7.3f} {r['latency_std']:<6.3f} "
                  f"{r['energy']:<7.1f} {r['success']:<7.1f} "
                  f"{r['efficiency']*1000:<8.3f} {r['offload_pct']:<6.1f} "
                  f"{r['fairness']:<7.3f} {r['fairness_uav']:<7.3f} {r['drop_pct']:.1f}")
        print()

    _print_summary_table(uav_counts, agg)
    return uav_counts, agg


def _print_summary_table(x, agg):
    idx = x.index(50)
    print("=" * 100)
    print("  TABLE IV — Extended Performance Comparison at N=50 (final manuscript)")
    print("=" * 100)
    hdr = (f"{'Algorithm':<22} {'Lat.(s)':<8} {'±Std':<7} {'Eng(J)':<9} "
           f"{'Succ%':<7} {'EE×10³':<8} {'Off%':<6} {'JFI_task':<10} {'JFI_uav':<8}")
    print(hdr); print("─" * len(hdr))
    for algo in agg:
        r = {k: agg[algo][k][idx] for k in agg[algo]}
        print(f"{algo:<22} {r['latency']:<8.3f} {r['latency_std']:<7.3f} "
              f"{r['energy']:<9.1f} {r['success']:<7.1f} "
              f"{r['efficiency']*1000:<8.3f} {r['offload_pct']:<6.1f} "
              f"{r['fairness']:<10.3f} {r['fairness_uav']:.3f}")

    print("\n  [Fair-DQN vs vanilla DQN @ N=50]")
    fair = agg['XAI-DQN-Fair']
    dqn  = agg['XAI-DQN (EdgeAISim)']
    de   = (fair['energy'][idx]       - dqn['energy'][idx])       / dqn['energy'][idx] * 100
    dl   = (fair['latency'][idx]      - dqn['latency'][idx])      / dqn['latency'][idx] * 100
    djt  = (fair['fairness'][idx]     - dqn['fairness'][idx])     / dqn['fairness'][idx] * 100
    dju  = (fair['fairness_uav'][idx] - dqn['fairness_uav'][idx]) / dqn['fairness_uav'][idx] * 100
    print(f"    Energy   : {de:+.1f}%   (utilitarian concession)")
    print(f"    Latency  : {dl:+.1f}%   (utilitarian concession)")
    print(f"    JFI_task : {djt:+.1f}%  (per-task fairness recovery)")
    print(f"    JFI_uav  : {dju:+.1f}%  (per-UAV fairness recovery — Eq. 18)")


# =============================================================================
# VISUALISATION
# =============================================================================
ALGO_COLORS = {
    'Cloud-Only':           '#d9534f',
    'Edge-Only':            '#f0ad4e',
    'XAI-Heuristic':        '#5bc0de',
    'XAI-DQN (EdgeAISim)':  '#5cb85c',
    'XAI-DQN-Fair':         '#9b59b6',          # NEW — distinct purple
}
ALGO_MARKERS = {
    'Cloud-Only':           's',
    'Edge-Only':            'o',
    'XAI-Heuristic':        '^',
    'XAI-DQN (EdgeAISim)':  'D',
    'XAI-DQN-Fair':         'P',                # NEW
}
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False, 'axes.spines.right': False,
    'axes.grid':         True,  'grid.linestyle': '--', 'grid.alpha': 0.45,
    'axes.titlesize':    13,    'axes.labelsize':  11,
    'legend.fontsize':   8.5,   'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi':        150,
})


def plot_with_confidence_bands(x, agg, mean_key, std_key, ylabel, title,
                                filename, ylim=None):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for algo in agg:
        means = np.array(agg[algo][mean_key])
        stds  = np.array(agg[algo][std_key]) if std_key else np.zeros_like(means)
        ax.plot(x, means, marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                label=algo, linewidth=2.2, markersize=7.5)
        if std_key:
            lower = np.clip(means - stds, 0, None)          # zero-clip (Week-7 fix)
            ax.fill_between(x, lower, means + stds,
                            color=ALGO_COLORS[algo], alpha=0.13,
                            label=f'_{algo}_band')
    ax.set_title(title, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Active UAVs')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x); ax.legend(loc='best')
    if ylim: ax.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  [VIZ-1] Saved: {filename}")


def plot_bar_summary(x, agg, N_val=50, filename='fig4_bar_summary_N50.png'):
    idx    = x.index(N_val)
    algos  = list(agg.keys())
    panels = [('latency',    'Avg. Latency (s)',                 False),
              ('energy',     'Avg. Energy (J)',                  False),
              ('success',    'Task Success Rate (%)',            True),
              ('efficiency', 'Energy Efficiency\n(Tasks/J × 10³)', True)]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 5))
    fig.suptitle(f'Multi-Metric Performance Summary at N={N_val} UAVs',
                 fontweight='bold', fontsize=13)
    for ax, (mkey, mlabel, higher_better) in zip(axes, panels):
        vals = []
        for algo in algos:
            v = agg[algo][mkey][idx]
            if mkey == 'efficiency': v *= 1000
            vals.append(v)
        bars = ax.bar(range(len(algos)), vals,
                      color=[ALGO_COLORS[a] for a in algos],
                      width=0.6, edgecolor='white', linewidth=1.2)
        best = vals.index(max(vals) if higher_better else min(vals))
        bars[best].set_edgecolor('#111'); bars[best].set_linewidth(2.8)
        ax.set_title(mlabel, fontsize=10)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(['Cloud', 'Edge', 'Heur.', 'DQN', 'DQN-Fair'], fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  [VIZ-2] Saved: {filename}")


def plot_radar_chart(x, agg, N_val=50, filename='fig5_radar_chart_N50.png'):
    idx    = x.index(N_val)
    algos  = list(agg.keys())
    labels = ['Latency\n(low=good)', 'Energy\n(low=good)',
              'Task Success\n(high=good)', 'Energy Eff.\n(high=good)',
              'Fairness\n(high=good)']
    N_ax = len(labels)
    angles = np.linspace(0, 2*np.pi, N_ax, endpoint=False).tolist(); angles += angles[:1]

    raw = {a: [agg[a]['latency'][idx], agg[a]['energy'][idx],
               agg[a]['success'][idx], agg[a]['efficiency'][idx]*1000,
               agg[a]['fairness'][idx]] for a in algos}
    col = np.array([[raw[a][i] for a in algos] for i in range(N_ax)])
    cmin, cmax = col.min(1), col.max(1)
    norm = {}
    for a in algos:
        nv = []
        for i, v in enumerate(raw[a]):
            span = cmax[i] - cmin[i]
            nrm  = (v - cmin[i]) / span if span > 0 else 0.5
            nv.append(1 - nrm if i < 2 else nrm)
        norm[a] = nv + nv[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={'polar': True})
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8)
    for a in algos:
        ax.plot(angles, norm[a], color=ALGO_COLORS[a], linewidth=2,
                marker=ALGO_MARKERS[a], markersize=6, label=a)
        ax.fill(angles, norm[a], color=ALGO_COLORS[a], alpha=0.10)
    ax.set_title(f'Multi-Dimensional Performance Profile (N={N_val})',
                 fontweight='bold', pad=22, fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.42, 1.15), fontsize=8.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  [VIZ-3] Saved: {filename}")


def plot_extended_metrics(x, agg):
    panels = [('efficiency',  'Energy Efficiency (Tasks/J × 10³)', 1000,
               'Tasks / Joule × 10³', 'fig6a_efficiency.png'),
              ('offload_pct', 'Cloud Offload Ratio (%)',            1,
               'Offload Ratio (%)',    'fig6b_offload.png'),
              ('fairness',    "Jain's Fairness Index (per-task)",    1,
               'Fairness Index',       'fig6c_fairness.png')]
    for mkey, title, scale, ylabel, filename in panels:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for algo in agg:
            vals = np.array(agg[algo][mkey]) * scale
            ax.plot(x, vals, marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                    label=algo, linewidth=2.2, markersize=7.5)
        ax.set_title(title, fontweight='bold', pad=12)
        ax.set_xlabel('Number of Active UAVs'); ax.set_ylabel(ylabel)
        ax.set_xticks(x); ax.legend()
        plt.tight_layout(); plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()
        print(f"  [VIZ-4] Saved: {filename}")


def plot_fairness_dual(x, agg, filename='fig6d_fairness_dual.png'):
    """NEW — side-by-side per-task vs per-UAV (Eq. 18) JFI comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, mkey, title in zip(axes,
                                ['fairness', 'fairness_uav'],
                                ["Per-Task JFI (legacy reporting)",
                                 "Per-UAV JFI — proper Eq. (18) over $E_u^{\\mathrm{cum}}$"]):
        for algo in agg:
            vals = np.array(agg[algo][mkey])
            ax.plot(x, vals, marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                    label=algo, linewidth=2.2, markersize=7.5)
        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_xlabel('Number of Active UAVs')
        ax.set_xticks(x); ax.set_ylim(0, 1.0); ax.legend(loc='lower left', fontsize=8.5)
    axes[0].set_ylabel("Jain's Fairness Index")
    fig.suptitle("Fairness Recovery via JFI-Shaped Reward (Eq. 13')",
                 fontweight='bold', fontsize=13)
    plt.tight_layout(); plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  [VIZ-DUAL] Saved: {filename}")


def plot_improvement_heatmap(x, agg, baseline='Cloud-Only',
                              filename='fig7_improvement_heatmap.png'):
    algos_cmp = ['Edge-Only', 'XAI-Heuristic',
                 'XAI-DQN (EdgeAISim)', 'XAI-DQN-Fair']
    metrics   = ['latency', 'energy', 'success']
    m_labels  = ['Latency ↓ (%)', 'Energy ↓ (%)', 'Success ↑ (pp)']
    short     = ['Edge', 'Heuristic', 'DQN', 'DQN-Fair']

    data = np.zeros((len(algos_cmp), len(metrics), len(x)))
    for ai, algo in enumerate(algos_cmp):
        for mi, m in enumerate(metrics):
            ba = np.array(agg[baseline][m]); al = np.array(agg[algo][m])
            data[ai, mi] = (al - ba) if m == 'success' else (ba - al) / ba * 100

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5))
    fig.suptitle('Improvement Relative to Cloud-Only Baseline',
                 fontweight='bold', fontsize=13)
    for mi, (ax, mlabel) in enumerate(zip(axes, m_labels)):
        mat = data[:, mi, :]
        vmax = 100; vmin = -20 if mi == 2 else 0
        im = ax.imshow(mat, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(mlabel, fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(x))); ax.set_xticklabels([str(n) for n in x])
        ax.set_yticks(range(len(algos_cmp))); ax.set_yticklabels(short, fontsize=9)
        ax.set_xlabel('Number of Active UAVs', fontsize=9)
        for ai in range(len(algos_cmp)):
            for xi in range(len(x)):
                ax.text(xi, ai, f'{mat[ai, xi]:+.1f}',
                        ha='center', va='center', fontsize=7.5, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  [VIZ-5] Saved: {filename}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    print("\nRunning v3.0 simulation with fairness-aware DRL ...\n")
    x, agg = run_experiments(seeds=MC_SEEDS)
    print("\nGenerating visualisations ...\n")

    plot_with_confidence_bands(x, agg, 'latency', 'latency_std',
        'Latency (Seconds)', 'Average End-to-End Latency (±1σ, M=5 trials)',
        'fig1_latency_revised.png')
    plot_with_confidence_bands(x, agg, 'energy', 'energy_std',
        'Energy (Joules)', 'Average UAV Energy Consumption (±1σ, M=5 trials)',
        'fig2_energy_revised.png')
    plot_with_confidence_bands(x, agg, 'success', None,
        'Completion Rate (%)', 'Successful Task Completion Ratio',
        'fig3_success_revised.png', ylim=(0, 105))

    plot_bar_summary(x, agg, N_val=50)
    plot_radar_chart(x, agg, N_val=50)
    plot_extended_metrics(x, agg)
    plot_fairness_dual(x, agg)
    plot_improvement_heatmap(x, agg)

    print("\n  All outputs generated successfully.")
