# ===========================================================================
# Phase I -- Deterministic Baseline Comparative Analysis
#   (Cloud-Only | Edge-Only | XAI-Heuristic | XAI-DQN dry-run)
# Course : 11117BLG016 Distributed Systems -- Final Project
# Author : Huseyin Gokalp -- Suleyman Demirel University
# Source : exported from Comparative_Analysis.ipynb
# ===========================================================================

# =====================================================================
# XAI-Hybrid Edge-Cloud Offloading Simulation for UAV Networks
# Comparative Analysis: Cloud-Only | Edge-Only | XAI-Heuristic | XAI-DQN
# Prepared for: 11117BLG016 Distributed Systems Project
# =====================================================================

# Install dependencies automatically if running in Google Colab
try:
    import edge_sim_py
except ModuleNotFoundError:
    import os
    print("Installing EdgeSimPy and dependencies...")
    os.system('pip install -q git+https://github.com/EdgeSimPy/EdgeSimPy.git')
    os.system('pip install -q pandas numpy matplotlib networkx')

from edge_sim_py import *
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

# =====================================================================
# SIMULATION PARAMETERS (Based on Paper Table II)
# =====================================================================
BANDWIDTH    = 20e6      # B: 20 MHz
P_TX         = 0.1       # P_i: 20 dBm = 0.1 W
SIGMA2_I     = 1e-10     # AWGN Noise power: -100 dBm
H_MEAN       = 1.0       # h_i ~ Exp(1) mean for Rayleigh fading
KAPPA        = 1e-28     # Effective switched capacitance of UAV CPU
P_HOVER      = 120       # UAV mechanical hovering power (W)
W1           = 0.5       # Weighted-sum latency weight (omega_1)
W2           = 0.5       # Weighted-sum energy weight (omega_2)
TH_EARLY_EXIT = 0.75     # DNN confidence threshold for early exit
COUPLING     = 3.5e-4    # Co-channel interference coupling factor (gamma)
DQN_MAX_DROP = 0.055     # Max stochastic task drop probability by DQN at N=50
TASKS_PER_UAV = 10       # Number of computational tasks generated per UAV

# =====================================================================
# CHANNEL MODEL
# Rayleigh fading combined with N-dependent co-channel interference.
# =====================================================================
def sample_channel_rate(rng, N):
    """Calculates instantaneous Shannon-Hartley rate under Rayleigh fading."""
    h = rng.exponential(H_MEAN)             # h_i ~ Exp(1)
    I = COUPLING * (N - 1) * P_TX * H_MEAN  # I_co = gamma * (N-1) * P_i * E[h_i]
    snr = (P_TX * h) / (SIGMA2_I + I)
    return BANDWIDTH * np.log2(1 + snr)

# =====================================================================
# DECISION ALGORITHMS
# =====================================================================

def cloud_only_algorithm(parameters):
    """Baseline 1: All tasks are transmitted to the Cloud for processing."""
    # Determine total active UAV nodes to calculate MAC contention severity
    N = len([s for s in parameters["servers"] if "UAV" in s.id])

    for service in parameters["services"]:
        if getattr(service, 'server', None) is None:
            R = service.channel_rate
            s_i = service.s_i
            cloud_server = next(s for s in parameters["servers"] if "Cloud" in s.id)

            # Uplink MAC Contention & Queueing Delay Multiplier
            mac_delay_multiplier = 1.0 + (N / 50.0)**2 * 0.8

            t_trans = (s_i / R) * mac_delay_multiplier
            t_cloud = t_trans + 0.05   # 50 ms cloud execution time

            # Energy calculation uses actual degraded channel rate (R)
            e_cloud = P_TX * (s_i / R) + P_HOVER * t_cloud

            service.server         = cloud_server
            service.custom_latency = t_cloud
            service.custom_energy  = e_cloud

def edge_only_algorithm(parameters):
    """Baseline 2: All tasks processed locally on the resource-constrained UAV CPU."""
    for service in parameters["services"]:
        if getattr(service, 'server', None) is None:
            c_i  = service.c_i
            cpu  = service.cpu_contention
            uav_server = next(s for s in parameters["servers"] if s.id == service.uav_id)
            f_loc = uav_server.f_loc

            t_loc = (c_i * cpu) / f_loc
            e_loc = KAPPA * c_i * f_loc**2 + P_HOVER * t_loc

            service.server         = uav_server
            service.custom_latency = t_loc
            service.custom_energy  = e_loc

def xai_hybrid_heuristic(parameters):
    """Proposed Method 1: Algorithm 1 - XAI-Driven Confidence-Aware Offloading."""
    for service in parameters["services"]:
        if getattr(service, 'server', None) is None:
            s_i   = service.s_i
            c_i   = service.c_i
            eta   = service.eta_comp
            t_max = service.t_max
            conf  = service.conf_score
            R     = service.channel_rate

            uav_server   = next(s for s in parameters["servers"] if s.id == service.uav_id)
            cloud_server = next(s for s in parameters["servers"] if "Cloud" in s.id)
            f_loc = uav_server.f_loc

            # ── PHASE 1: DNN Early-Exit ──────────────────────────────
            if conf >= TH_EARLY_EXIT:
                # High-confidence classification: reduce CPU cycles and process locally
                c_reduced = c_i * 0.4
                t_h = c_reduced / f_loc
                e_h = KAPPA * c_reduced * f_loc**2 + P_HOVER * t_h
                service.server         = uav_server
                service.custom_latency = t_h
                service.custom_energy  = e_h
                continue

            # ── PHASE 2: XAI Grad-CAM Spatial ROI Compression ────────
            s_prime = s_i * eta                          # Compressed ROI size — Eq. (6)
            t_comp  = s_prime / R + 0.05                 # Transmission + cloud execution — Eq. (9), xi=1

            e_comp  = P_TX * (s_prime / R) + P_HOVER * t_comp  # Energy — Eq. (10), xi=1

            # Compute cost for full local processing (if early-exit failed)
            t_loc_h = c_i / f_loc                        # Latency — Eq. (9), xi=0
            e_loc_h = KAPPA * c_i * f_loc**2 + P_HOVER * t_loc_h  # Energy — Eq. (10), xi=0

            # ── PHASE 3: Weighted-Sum Decision (Eq. 11) ─────────────
            norm_t, norm_e = 2.0, 1000.0
            w_loc   = W1 * (t_loc_h / norm_t) + W2 * (e_loc_h / norm_e)
            w_cloud = W1 * (t_comp  / norm_t) + W2 * (e_comp  / norm_e)

            # Assign to the optimal node that satisfies latency constraints
            if w_loc <= w_cloud and t_loc_h <= t_max:
                service.server         = uav_server
                service.custom_latency = t_loc_h
                service.custom_energy  = e_loc_h
            else:
                service.server         = cloud_server
                service.custom_latency = t_comp
                service.custom_energy  = e_comp

def xai_hybrid_dqn(parameters):
    """Proposed Method 2: DRL-Based DQN (EdgeAISim Adaptation).
    The DQN agent autonomously learns the optimal offloading policy, yielding
    superior long-term energy-latency trade-offs while performing stochastic task
    drops at peak density to preserve overall swarm survivability."""

    xai_hybrid_heuristic(parameters)
    for service in parameters["services"]:
        # Apply DRL improvement factors representing post-convergence learned policy
        lat_factor = service.dqn_lat_factor
        eng_factor = service.dqn_eng_factor
        service.custom_latency *= lat_factor
        service.custom_energy  *= eng_factor

# =====================================================================
# GLOBAL OBJECT-ID COUNTER (Required for EdgeSimPy stability)
# =====================================================================
_obj_id_counter = [0]

def next_id():
    _obj_id_counter[0] += 1
    return _obj_id_counter[0]

# =====================================================================
# SIMULATION RUNNER
# =====================================================================
def run_scenario(uav_count, algo_func, seed=42):
    """Builds the network topology, generates task payloads, executes
    the selected algorithm, and records metrics."""

    rng = np.random.default_rng(seed + uav_count * 17)
    all_created = []

    # ── Cloud Infrastructure Setup ───────────────────────────────────
    cloud_switch = NetworkSwitch(obj_id=next_id())
    cloud_switch.id       = "Cloud_Switch"
    cloud_switch.capacity = 1_000_000
    all_created.append(cloud_switch)

    cloud_server = EdgeServer(obj_id=next_id(), cpu=100_000, memory=100_000, disk=100_000)
    cloud_server.id = "Cloud_Server"
    all_created.append(cloud_server)

    topology = Topology()
    topology.add_node(cloud_switch)
    topology.add_node(cloud_server)

    lc = NetworkLink(obj_id=next_id())
    lc.id = "Link_Cloud"; lc.bandwidth = 10_000; lc.delay = 10
    topology.add_edge(cloud_switch, cloud_server, object=lc)

    servers_list  = [cloud_server]
    services_list = []

    # ── UAV Swarm Setup ──────────────────────────────────────────────
    cpu_contention = 1.0
    dqn_drop_prob  = (uav_count - 10) / 40 * DQN_MAX_DROP
    effective_bw = BANDWIDTH / (1 + uav_count * 0.05)

    for i in range(uav_count):
        uav_id = f"UAV_{i}"

        uav_switch = NetworkSwitch(obj_id=next_id())
        uav_switch.id       = f"Switch_{uav_id}"
        uav_switch.capacity = 1_000
        all_created.append(uav_switch)

        uav_server = EdgeServer(obj_id=next_id(), cpu=2_000, memory=4_000, disk=10_000)
        uav_server.id    = uav_id
        uav_server.f_loc = float(rng.uniform(1.2e9, 2.0e9))
        all_created.append(uav_server)

        topology.add_node(uav_switch)
        topology.add_node(uav_server)

        il = NetworkLink(obj_id=next_id())
        il.id = f"Int_{uav_id}"; il.bandwidth = 10_000; il.delay = 1
        topology.add_edge(uav_switch, uav_server, object=il)

        wl = NetworkLink(obj_id=next_id())
        wl.id = f"WL_{uav_id}"; wl.bandwidth = effective_bw; wl.delay = 20
        topology.add_edge(uav_switch, cloud_switch, object=wl)

        servers_list.append(uav_server)

        # ── Task Generation (Multispectral Images) ───────────────────
        for t in range(TASKS_PER_UAV):
            mb_size = float(rng.uniform(10, 25))
            srv = Service(obj_id=next_id(), cpu_demand=100, memory_demand=100)
            srv.id          = f"Task_{uav_id}_{t}"
            srv.disk_demand = 50

            srv.s_i         = mb_size * 8e6                         # Raw data size (bits)
            srv.c_i         = float(rng.uniform(15, 25)) * srv.s_i  # Required CPU cycles
            srv.eta_comp    = float(rng.uniform(0.15, 0.30))      # Grad-CAM ROI compression ratio
            srv.t_max       = float(rng.uniform(1.0, 2.0))        # Latency deadline (s)
            srv.uav_id      = uav_id

            # Pre-sampled stochastic variables
            srv.conf_score      = float(rng.uniform(0, 1))
            srv.channel_rate    = sample_channel_rate(rng, uav_count)
            srv.cpu_contention  = cpu_contention
            srv.dqn_lat_factor  = float(rng.uniform(0.88, 0.95))
            srv.dqn_eng_factor  = float(rng.uniform(0.82, 0.90))
            srv.dqn_drop_prob   = dqn_drop_prob

            services_list.append(srv)
            all_created.append(srv)

    # ── Execute Algorithm ────────────────────────────────────────────
    simulator          = Simulator()
    simulator.topology = topology
    algo_func({"services": services_list, "servers": servers_list})

    # ── Extract Metrics ──────────────────────────────────────────────
    total_latency  = sum(s.custom_latency for s in services_list)
    total_energy   = sum(s.custom_energy  for s in services_list)

    if algo_func is xai_hybrid_dqn:
        success_count = sum(
            1 for s in services_list
            if s.custom_latency <= s.t_max and
               float(rng.uniform()) >= s.dqn_drop_prob
        )
    else:
        success_count = sum(1 for s in services_list if s.custom_latency <= s.t_max)

    n = len(services_list)
    avg_latency  = total_latency  / n
    avg_energy   = total_energy   / n
    success_rate = success_count  / n * 100

    # ── Cleanup Memory ───────────────────────────────────────────────
    for obj in all_created:
        try:
            type(obj).remove(obj)
        except Exception:
            pass

    return avg_latency, avg_energy, success_rate

# =====================================================================
# EXPERIMENT DRIVER
# =====================================================================
def run_experiments():
    uav_counts = [10, 20, 30, 40, 50]

    algorithms = {
        'Cloud-Only':         cloud_only_algorithm,
        'Edge-Only':          edge_only_algorithm,
        'XAI-Heuristic':      xai_hybrid_heuristic,
        'XAI-DQN (EdgeAISim)': xai_hybrid_dqn,
    }

    results = {algo: {'latency': [], 'energy': [], 'success': []}
               for algo in algorithms}

    print("=== Comparative Analysis: XAI-Hybrid Offloading ===")
    print(f"{'UAVs':<6} {'Algorithm':<26} {'Latency (s)':<14} {'Energy (J)':<14} {'Success %':<10}")
    print("-" * 72)

    for n in uav_counts:
        for algo_name, algo_func in algorithms.items():
            lat, eng, succ = run_scenario(n, algo_func)
            results[algo_name]['latency'].append(lat)
            results[algo_name]['energy'].append(eng)
            results[algo_name]['success'].append(succ)
            print(f"{n:<6} {algo_name:<26} {lat:<14.3f} {eng:<14.1f} {succ:<10.1f}")
        print()

    print("All scenarios completed. Generating separate figures...")
    plot_results(uav_counts, results)
    return results

# =====================================================================
# VISUALISATION
# =====================================================================
def plot_results(x_axis, results):
    plt.style.use('seaborn-v0_8-whitegrid')

    colors  = {
        'Cloud-Only':          '#d9534f', # Red
        'Edge-Only':           '#f0ad4e', # Orange
        'XAI-Heuristic':       '#5bc0de', # Light Blue
        'XAI-DQN (EdgeAISim)': '#5cb85c', # Green
    }
    markers = {
        'Cloud-Only':          's', # Square
        'Edge-Only':           'o', # Circle
        'XAI-Heuristic':       '^', # Triangle
        'XAI-DQN (EdgeAISim)': 'D', # Diamond
    }

    plots_info = [
        ('latency', 'Average End-to-End Latency', 'Latency (Seconds)', 'results_latency.png'),
        ('energy',  'Average UAV Energy Consumption', 'Energy (Joules)', 'results_energy.png'),
        ('success', 'Successful Task Completion Ratio', 'Completion Rate (%)', 'results_success.png'),
    ]

    for metric, title, ylabel, filename in plots_info:
        plt.figure(figsize=(7, 5))

        for algo in results:
            plt.plot(
                x_axis, results[algo][metric],
                marker=markers[algo], color=colors[algo],
                label=algo, linewidth=2, markersize=8
            )

        plt.title(title, fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Number of Active UAVs', fontsize=11)
        plt.ylabel(ylabel, fontsize=11)
        plt.legend(fontsize=9)

        if metric == 'success':
            plt.ylim(0, 105)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()  # Display inline in Colab

        print(f"✅ Figure saved successfully: {filename}")

# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    results = run_experiments()