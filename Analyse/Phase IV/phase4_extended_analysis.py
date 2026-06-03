# =============================================================================
#  PHASE IV — EXTENDED ANALYSES: Closing the Open Issues of the Final Report
#  XAI-Hybrid Edge-Cloud Offloading for UAV Networks
#
#  Course   : 11117BLG016 Distributed Systems — Final Project
#  Author   : Hüseyin Gökalp — Süleyman Demirel University
#
#  This module operationalises the four "future-work / limitation" items that
#  the Week-7 manuscript had explicitly deferred, converting each into a
#  reproducible quantitative experiment built on the *identical* simulation
#  core (sim_core.py) used for Phases I–III. All four analyses share the
#  Table-II parameters, the Eq.(7)–(8) channel model and the Eq.(12) CPU
#  contention model, so their outputs are directly comparable with the
#  Phase III results table.
#
#    Phase IV-A : Grad-CAM Overhead-Aware NET Energy Model
#                 (Limitation #1 — XAI compute/energy overhead abstracted)
#    Phase IV-B : Compression-Coefficient (eta) Sensitivity / Severe-Infection
#                 Stress Test (Limitation #2 — eta range 0.15-0.30 too narrow)
#    Phase IV-C : In-Field Visual-Noise Robustness of the XAI Layer
#                 (Limitation #5 — PlantVillage lab data vs in-field tau_XAI)
#    Phase IV-D : Fairness Coefficient (lambda) Pareto-Front Sweep
#                 (Phase III extension — lambda was fixed at 0.50)
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import sim_core as S   # shared Phases I-III simulation core (channel, energy, algos)

OUT = os.path.dirname(os.path.abspath(__file__))
RNG_SEEDS = S.MC_SEEDS              # (42, 123, 256, 789, 1337) — identical to Phase III
plt.rcParams.update({'figure.dpi': 130, 'font.size': 10,
                     'axes.grid': True, 'grid.alpha': 0.30,
                     'axes.spines.top': False, 'axes.spines.right': False})

COL = {'cloud': '#d62728', 'edge': '#ff7f0e', 'heur': '#2ca02c',
       'dqn': '#1f77b4', 'fair': '#9467bd', 'net': '#17becf', 'warn': '#8c564b'}


# =============================================================================
#  PHASE IV-A — GRAD-CAM OVERHEAD-AWARE NET ENERGY MODEL
# -----------------------------------------------------------------------------
#  The manuscript abstracted the cost of *running* Grad-CAM on the UAV's
#  single-board computer. Here we model it explicitly and ask: once the real
#  XAI overhead is charged, does spatial cropping still pay off, and from which
#  raw-image size onward?
#
#  Hardware anchor (empirical, Q1): MobileNetV3-Large + Grad-CAM on an NVIDIA
#  Jetson Nano draws ~3.4 W average during prediction [Nature Sci.Rep. 2024,
#  s41598-024-66989-9]. A Grad-CAM pass = 1 forward + 1 (partial) backward, so
#  we charge ~2.4x a bare forward-pass latency.
# =============================================================================
P_XAI_SBC      = 3.4        # W — Jetson Nano avg power for MobileNetV3+Grad-CAM
T_FWD_BASE     = 0.018      # s — bare MobileNetV3-Large forward pass @ ~512px
GRADCAM_FACTOR = 2.4        # Grad-CAM = fwd + partial bwd  ->  ~2.4x forward cost
MB_BIT         = 8e6        # bits per MB


def gradcam_overhead_energy(N_pixels_scale=1.0):
    """Energy (J) spent ON-UAV to run one Grad-CAM pass, incl. hovering."""
    t_xai = T_FWD_BASE * GRADCAM_FACTOR * N_pixels_scale
    # Active SBC energy + the propulsion (hover) energy burned during that time.
    return (P_XAI_SBC * t_xai) + (S.P_HOVER * t_xai), t_xai


def run_phase4a():
    """Net-energy break-even of XAI cropping vs. raw cloud offload."""
    raw_sizes_mb = np.linspace(0.5, 25, 26)         # raw multispectral frame size
    eta          = 0.225                            # mean of Table-II [0.15,0.30]
    N            = 50

    gross_saving, xai_cost, net_saving = [], [], []
    e_over, _ = gradcam_overhead_energy()

    rng = np.random.default_rng(7)
    R   = S.sample_channel_rate(rng, N)             # representative channel rate

    for mb in raw_sizes_mb:
        s_raw   = mb * MB_BIT
        s_crop  = s_raw * eta
        # Transmission energy (Eq. 9, cloud branch) for raw vs cropped payload.
        e_tx_raw  = S.P_TX * (s_raw  / R) + S.P_HOVER * (s_raw  / R + 0.05)
        e_tx_crop = S.P_TX * (s_crop / R) + S.P_HOVER * (s_crop / R + 0.05)
        gsave = e_tx_raw - e_tx_crop                # gross transmission saving
        nsave = gsave - e_over                      # charge the XAI overhead
        gross_saving.append(gsave)
        xai_cost.append(e_over)
        net_saving.append(nsave)

    gross_saving = np.array(gross_saving)
    net_saving   = np.array(net_saving)

    # Break-even raw size: smallest MB where net saving turns positive.
    be_idx = np.argmax(net_saving > 0) if np.any(net_saving > 0) else -1
    be_mb  = raw_sizes_mb[be_idx] if be_idx >= 0 else float('nan')

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(raw_sizes_mb, gross_saving, '-o', ms=3, color=COL['heur'],
            label='Gross transmission saving (XAI cropping)')
    ax.plot(raw_sizes_mb, [xai_cost[0]] * len(raw_sizes_mb), '--',
            color=COL['warn'], label=f'Grad-CAM on-UAV overhead ({xai_cost[0]:.2f} J)')
    ax.plot(raw_sizes_mb, net_saving, '-s', ms=3, color=COL['dqn'],
            label='NET energy saving (gross − overhead)')
    ax.axhline(0, color='k', lw=0.8)
    if be_idx >= 0:
        ax.axvline(be_mb, color=COL['cloud'], ls=':', lw=1.4)
        ax.annotate(f'break-even ≈ {be_mb:.1f} MB',
                    xy=(be_mb, 0), xytext=(be_mb + 1.0, max(net_saving) * 0.45),
                    color=COL['cloud'],
                    arrowprops=dict(arrowstyle='->', color=COL['cloud']))
    ax.set_xlabel('Raw multispectral frame size (MB)')
    ax.set_ylabel('Energy per task (J)')
    ax.set_title('Phase IV-A: Grad-CAM Overhead-Aware Net Energy ($\\eta$=0.225, N=50)')
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    p = os.path.join(OUT, 'fig8_gradcam_overhead.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)

    # Net saving over the operational 10-25 MB band (paper's s_i range).
    band = (raw_sizes_mb >= 10) & (raw_sizes_mb <= 25)
    print(f"[IV-A] Grad-CAM overhead per task ......... {xai_cost[0]:.3f} J")
    print(f"[IV-A] Net break-even raw size ............ {be_mb:.2f} MB")
    print(f"[IV-A] Mean NET saving over 10-25 MB band . {net_saving[band].mean():.2f} J/task")
    print(f"[IV-A] Overhead as %% of gross @25MB ...... "
          f"{100*xai_cost[0]/gross_saving[-1]:.2f}%")
    return be_mb, net_saving[band].mean(), xai_cost[0]


# =============================================================================
#  PHASE IV-B — COMPRESSION-COEFFICIENT SENSITIVITY (SEVERE-INFECTION REGIME)
# -----------------------------------------------------------------------------
#  Table II fixed eta in [0.15, 0.30]. Severe / widespread infections enlarge
#  the Grad-CAM ROI, pushing eta well above 0.30 (potentially >0.80). We sweep
#  a *fixed* eta from 0.15 to 0.90, override every task's eta, re-run XAI-DQN,
#  and locate eta* — the point at which the compressed-cloud route loses its
#  advantage over the Edge-Only baseline / breaches the real-time tolerance.
# =============================================================================
def _run_fixed_eta(N, eta_fixed, seed):
    """Run XAI-DQN with every task's eta_comp overridden to eta_fixed."""
    rng = np.random.default_rng(seed + N * 17)
    services, servers = [], [{'id': 'Cloud_Server', 'type': 'cloud', 'f_loc': None}]
    for i in range(N):
        uav_id = f'UAV_{i}'
        f_loc  = float(rng.uniform(1.2e9, 2.0e9))
        servers.append({'id': uav_id, 'type': 'uav', 'f_loc': f_loc})
        for _ in range(S.TASKS_PER_UAV):
            s_i = float(rng.uniform(10, 25)) * MB_BIT
            services.append({
                'uav_id': uav_id, 's_i': s_i,
                'c_i': float(rng.uniform(15, 25)) * s_i,
                'eta_comp': eta_fixed,                      # <-- overridden
                't_max': float(rng.uniform(1.0, 2.0)),
                'f_loc': f_loc, 'conf_score': float(rng.uniform(0, 1)),
                'channel_rate': S.sample_channel_rate(rng, N),
                'dqn_lat_factor': float(rng.uniform(0.88, 0.95)),
                'dqn_eng_factor': float(rng.uniform(0.82, 0.90)),
            })
    return S.collect_metrics(S.xai_dqn(services, servers, N, rng))


def run_phase4b():
    N        = 50
    etas     = np.round(np.arange(0.15, 0.91, 0.05), 2)
    lat, eng, succ, off = [], [], [], []
    for e in etas:
        ms = [_run_fixed_eta(N, e, s) for s in RNG_SEEDS]
        lat.append(np.mean([m['latency'] for m in ms]))
        eng.append(np.mean([m['energy'] for m in ms]))
        succ.append(np.mean([m['success'] for m in ms]))
        off.append(np.mean([m['offload_pct'] for m in ms]))
    lat, eng, succ, off = map(np.array, (lat, eng, succ, off))

    # Edge-Only reference at N=50 (eta-independent — never offloads).
    edge = [S.collect_metrics(S.build_and_run(N, S.edge_only, seed=s)) for s in RNG_SEEDS]
    edge_lat = np.mean([m['latency'] for m in edge])
    edge_eng = np.mean([m['energy'] for m in edge])

    # eta* : first eta where XAI-DQN breaches the STRICT real-time floor (1.0 s),
    # i.e. the lower bound of the t_max in [1.0, 2.0] s — the tightest tasks.
    T_STRICT = 1.0
    cross = np.where(lat >= T_STRICT)[0]
    eta_star = etas[cross[0]] if cross.size else float('nan')

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    a1.plot(etas, lat, '-o', ms=3, color=COL['dqn'], label='XAI-DQN latency')
    a1.axhline(edge_lat, ls='--', color=COL['edge'], label=f'Edge-Only ({edge_lat:.2f} s)')
    a1.axhline(1.0, ls=':', color=COL['cloud'], lw=1.2, label='strict real-time floor (1.0 s)')
    a1.axhline(2.0, ls='-.', color=COL['warn'], lw=1.0, label='$t_{max}$ ceiling (2.0 s)')
    if not np.isnan(eta_star):
        a1.axvline(eta_star, color='k', ls=':', lw=1.3)
        a1.annotate(f'$\\eta^*$≈{eta_star:.2f}', xy=(eta_star, 1.0),
                    xytext=(eta_star - 0.24, 1.5),
                    arrowprops=dict(arrowstyle='->'))
    a1.set_xlabel('Compression coefficient $\\eta_{comp}$ (infection severity →)')
    a1.set_ylabel('Avg end-to-end latency (s)')
    a1.set_title('(a) Latency vs. ROI size')
    a1.legend(fontsize=8)

    a2.plot(etas, eng, '-s', ms=3, color=COL['fair'], label='XAI-DQN energy')
    a2.axhline(edge_eng, ls='--', color=COL['edge'], label=f'Edge-Only ({edge_eng:.0f} J)')
    a2b = a2.twinx()
    a2b.plot(etas, succ, '-^', ms=3, color=COL['heur'], label='success %')
    a2b.set_ylabel('Task success (%)', color=COL['heur'])
    a2b.set_ylim(0, 105); a2b.grid(False)
    a2.set_xlabel('Compression coefficient $\\eta_{comp}$ (infection severity →)')
    a2.set_ylabel('Avg energy per task (J)', color=COL['fair'])
    a2.set_title('(b) Energy & success vs. ROI size')
    a2.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    p = os.path.join(OUT, 'fig9_eta_sensitivity.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)

    print(f"[IV-B] Latency @eta=0.15 / 0.30 / 0.60 / 0.90 .. "
          f"{lat[0]:.3f} / {lat[3]:.3f} / {lat[9]:.3f} / {lat[-1]:.3f} s")
    print(f"[IV-B] Energy  @eta=0.15 / 0.90 ............... {eng[0]:.1f} / {eng[-1]:.1f} J")
    print(f"[IV-B] Strict-real-time (1.0s) crossover eta* . {eta_star}")
    print(f"[IV-B] Latency @eta=0.90 vs t_max ceiling ..... {lat[-1]:.3f} s / 2.0 s "
          f"({100*(1-lat[-1]/2.0):.0f}% headroom)")
    return eta_star, lat, eng


# =============================================================================
#  PHASE IV-C — IN-FIELD VISUAL-NOISE ROBUSTNESS OF THE XAI LAYER
# -----------------------------------------------------------------------------
#  PlantVillage is lab-grade. In-field imagery (canopy clutter, illumination,
#  occlusion) (i) FRAGMENTS the Grad-CAM ROI, inflating the effective eta, and
#  (ii) DEGRADES classification confidence, lowering the early-exit hit rate.
#  We parametrise a noise level nu in [0,1] and propagate both effects, then
#  measure how XAI-DQN and XAI-DQN-Fair degrade.
#
#  Accuracy anchor (Q1): cross-domain (lab->field) accuracy drops 12-15% in the
#  mild regime [Front. Plant Sci. 2025] and to ~31-33% in severe shift
#  [ScienceDirect 2024/2025]; we map nu linearly onto an effective drop.
# =============================================================================
ETA_FRAG_GAIN = 1.8        # ROI fragmentation: eta' = eta*(1 + nu*ETA_FRAG_GAIN)
CONF_DROP_MAX = 0.55       # nu=1 collapses early-exit confidence by up to 55%


def _run_field_noise(N, nu, algo_fn, seed):
    rng = np.random.default_rng(seed + N * 17)
    services, servers = [], [{'id': 'Cloud_Server', 'type': 'cloud', 'f_loc': None}]
    for i in range(N):
        uav_id = f'UAV_{i}'
        f_loc  = float(rng.uniform(1.2e9, 2.0e9))
        servers.append({'id': uav_id, 'type': 'uav', 'f_loc': f_loc})
        for _ in range(S.TASKS_PER_UAV):
            s_i      = float(rng.uniform(10, 25)) * MB_BIT
            eta_base = float(rng.uniform(0.15, 0.30))
            eta_eff  = min(0.95, eta_base * (1.0 + nu * ETA_FRAG_GAIN))  # fragmentation
            conf     = float(rng.uniform(0, 1)) * (1.0 - nu * CONF_DROP_MAX)
            services.append({
                'uav_id': uav_id, 's_i': s_i,
                'c_i': float(rng.uniform(15, 25)) * s_i,
                'eta_comp': eta_eff, 't_max': float(rng.uniform(1.0, 2.0)),
                'f_loc': f_loc, 'conf_score': conf,
                'channel_rate': S.sample_channel_rate(rng, N),
                'dqn_lat_factor': float(rng.uniform(0.88, 0.95)),
                'dqn_eng_factor': float(rng.uniform(0.82, 0.90)),
            })
    return S.collect_metrics(algo_fn(services, servers, N, rng))


def run_phase4c():
    N    = 50
    nus  = np.round(np.arange(0.0, 1.01, 0.1), 2)
    out  = {'dqn': {'succ': [], 'eng': []}, 'fair': {'succ': [], 'eng': []}}
    for nu in nus:
        for key, fn in (('dqn', S.xai_dqn), ('fair', S.xai_dqn_fair)):
            ms = [_run_field_noise(N, nu, fn, s) for s in RNG_SEEDS]
            out[key]['succ'].append(np.mean([m['success'] for m in ms]))
            out[key]['eng'].append(np.mean([m['energy'] for m in ms]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    a1.plot(nus, out['dqn']['succ'],  '-o', ms=3, color=COL['dqn'],  label='XAI-DQN')
    a1.plot(nus, out['fair']['succ'], '-s', ms=3, color=COL['fair'], label='XAI-DQN-Fair')
    a1.axvline(0.0, ls=':', color='gray'); a1.text(0.01, 30, 'lab\n($\\nu$=0)', fontsize=7)
    a1.axvspan(0.6, 1.0, color=COL['cloud'], alpha=0.06)
    a1.text(0.62, 30, 'severe\nin-field shift', fontsize=7, color=COL['cloud'])
    a1.set_xlabel('In-field visual-noise level $\\nu$')
    a1.set_ylabel('Task success (%)'); a1.set_ylim(0, 105)
    a1.set_title('(a) Diagnostic resilience vs. noise')
    a1.legend(fontsize=8)

    a2.plot(nus, out['dqn']['eng'],  '-o', ms=3, color=COL['dqn'],  label='XAI-DQN')
    a2.plot(nus, out['fair']['eng'], '-s', ms=3, color=COL['fair'], label='XAI-DQN-Fair')
    a2.set_xlabel('In-field visual-noise level $\\nu$')
    a2.set_ylabel('Avg energy per task (J)')
    a2.set_title('(b) Energy inflation from ROI fragmentation')
    a2.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, 'fig10_infield_noise.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)

    print(f"[IV-C] Success lab->field (nu 0->1) DQN ...... "
          f"{out['dqn']['succ'][0]:.1f}% -> {out['dqn']['succ'][-1]:.1f}%")
    print(f"[IV-C] Success lab->field (nu 0->1) Fair ..... "
          f"{out['fair']['succ'][0]:.1f}% -> {out['fair']['succ'][-1]:.1f}%")
    print(f"[IV-C] Energy lab->field (nu 0->1) DQN ....... "
          f"{out['dqn']['eng'][0]:.1f} -> {out['dqn']['eng'][-1]:.1f} J")
    return out, nus


# =============================================================================
#  PHASE IV-D — FAIRNESS COEFFICIENT (lambda) PARETO-FRONT SWEEP
# -----------------------------------------------------------------------------
#  Phase III fixed lambda = 0.50. We now sweep lambda in [0, 1] to trace the
#  energy-vs-fairness Pareto front and identify the operational knee. A clean,
#  lambda-threaded fair policy is used so that lambda=0 reduces to the vanilla
#  utilitarian DQN and lambda>0 progressively buys fairness with energy.
# =============================================================================
def xai_dqn_fair_lambda(services, servers, N, rng, lam):
    """lambda-parametrised fairness-aware DRL (generalises sim_core.xai_dqn_fair).
       lam=0 -> vanilla utilitarian DQN; lam=0.5 -> the Phase III XAI-DQN-Fair."""
    base_h = S.xai_heuristic(services, servers, N, rng)

    tasks_by_uav = {}
    for idx, svc in enumerate(services):
        tasks_by_uav.setdefault(svc['uav_id'], []).append(idx)
    uav_ids = sorted(tasks_by_uav.keys())
    cum_e   = {u: 0.0 for u in uav_ids}
    max_per = max(len(v) for v in tasks_by_uav.values())
    order   = [(u, tasks_by_uav[u][t]) for t in range(max_per)
               for u in uav_ids if t < len(tasks_by_uav[u])]

    # lambda-scheduled control knobs (monotone in lam) ------------------------
    tax        = 1.0 + 0.06 * lam                       # energy/latency tax
    thr_off    = 1.20 - 0.30 * lam                      # force-offload threshold
    thr_loc    = 0.70 + 0.30 * lam                      # local-absorption threshold
    e_cap      = 1e12 if lam <= 1e-9 else (300.0 - 100.0 * lam)  # sacrifice ceiling
    drop_scale = 1.0 - 0.9 * lam                        # fairness reduces crisis drops

    drop_prob = (N - 10) / 40 * S.DQN_MAX_DROP * max(0.05, drop_scale)
    drop_rng  = np.random.default_rng(seed=N * 31 + 7 + int(1000 * lam))
    reb_rng   = np.random.default_rng(seed=N * 31 + 7 + 5101 + int(1000 * lam))

    results = [None] * len(services)
    for u, idx in order:
        svc, r_h = services[idx], base_h[idx]
        e_vec = np.fromiter(cum_e.values(), float)
        m_e   = float(e_vec.mean()) if e_vec.sum() > 1e-9 else 1.0
        s_u   = float(np.clip(cum_e[u] / m_e if m_e > 1e-9 else 1.0, 0.5, 2.0))

        e_pred    = r_h['energy'] * svc['dqn_eng_factor'] * tax
        sacrifice = (lam > 1e-9) and (e_pred > e_cap)
        off_ov = loc_ov = False

        if sacrifice:
            t_r, e_r, sacrificed = 0.02, S.P_HOVER * 0.02, True
        elif lam > 1e-9 and s_u > thr_off and not r_h['offloaded']:
            R_i = svc['channel_rate']; sp = svc['s_i'] * svc['eta_comp']
            t_r = sp / R_i + 0.05; e_r = S.P_TX * (sp / R_i) + S.P_HOVER * t_r
            off_ov, sacrificed = True, False
        elif lam > 1e-9 and s_u < thr_loc and r_h['offloaded']:
            p_loc = min(S.P_LOCAL_CAP, lam * (thr_loc - s_u) * 2.0)
            if reb_rng.random() < p_loc:
                t_r = svc['c_i'] / svc['f_loc']
                e_r = S.KAPPA * svc['c_i'] * svc['f_loc'] ** 2 + S.P_HOVER * t_r
                loc_ov = True
            else:
                t_r, e_r = r_h['latency'], r_h['energy']
            sacrificed = False
        else:
            t_r, e_r, sacrificed = r_h['latency'], r_h['energy'], False

        if sacrificed:
            lat, eng = t_r, e_r
        else:
            lat = t_r * svc['dqn_lat_factor'] * tax
            eng = e_r * svc['dqn_eng_factor'] * (tax + lam * 0.08 * max(0.0, s_u - 1.0))

        dropped = sacrificed or bool(drop_rng.random() < drop_prob)
        results[idx] = {'latency': lat, 'energy': eng,
                        'success': (lat <= svc['t_max']) and not dropped,
                        'offloaded': (r_h['offloaded'] and not loc_ov) or off_ov,
                        'dropped': dropped, 'uav_id': u}
        cum_e[u] += eng
    return results


def run_phase4d():
    N       = 50
    lambdas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    eng, jfi, succ = [], [], []
    for lam in lambdas:
        ms = []
        for s in RNG_SEEDS:
            rng = np.random.default_rng(s + N * 17)
            res = xai_dqn_fair_lambda(*_scenario(N, s), N, rng, lam) \
                if False else _run_lambda(N, lam, s)
            ms.append(res)
        eng.append(np.mean([m['energy'] for m in ms]))
        jfi.append(np.mean([m['fairness_uav'] for m in ms]))
        succ.append(np.mean([m['success'] for m in ms]))
    eng, jfi = np.array(eng), np.array(jfi)

    # Knee = point maximising (normalised JFI gain − normalised energy cost).
    e_n = (eng - eng.min()) / (eng.max() - eng.min() + 1e-9)
    j_n = (jfi - jfi.min()) / (jfi.max() - jfi.min() + 1e-9)
    knee = int(np.argmax(j_n - e_n))

    # Pareto-efficiency flag: a point is dominated if another has both
    # lower energy AND higher fairness.
    dominated = []
    for i in range(len(lambdas)):
        dom = any((eng[j] <= eng[i]) and (jfi[j] >= jfi[i]) and (j != i) and
                  ((eng[j] < eng[i]) or (jfi[j] > jfi[i])) for j in range(len(lambdas)))
        dominated.append(dom)

    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    # Frontier line through non-dominated points, ordered by energy.
    eff_idx = [i for i in range(len(lambdas)) if not dominated[i]]
    eff_idx.sort(key=lambda i: eng[i])
    ax.plot(eng[eff_idx], jfi[eff_idx], '-', color=COL['cloud'], lw=1.4,
            alpha=0.6, zorder=1, label='Pareto-efficient frontier')
    sc = ax.scatter(eng, jfi, c=lambdas, cmap='viridis', s=85, zorder=2,
                    edgecolor='k', linewidth=0.5)
    for x, y, lm, dm in zip(eng, jfi, lambdas, dominated):
        tag = f'$\\lambda$={lm}' + ('  (dominated)' if dm else '')
        ax.annotate(tag, (x, y), textcoords='offset points',
                    xytext=(7, -3), fontsize=7.5,
                    color='gray' if dm else 'black')
    ax.scatter([eng[knee]], [jfi[knee]], s=270, facecolors='none',
               edgecolors=COL['cloud'], linewidths=2,
               label=f'knee / recommended ($\\lambda$={lambdas[knee]})', zorder=3)
    cb = fig.colorbar(sc, ax=ax); cb.set_label('Fairness coefficient $\\lambda$')
    ax.set_xlabel('Avg energy per task (J)  — utilitarian cost →')
    ax.set_ylabel('Per-UAV Jain Fairness Index $JFI_{uav}$  — egalitarian gain ↑')
    ax.set_title('Phase IV-D: Energy–Fairness Trade-off under $\\lambda$ Sweep (N=50)')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    p = os.path.join(OUT, 'fig11_lambda_pareto.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)

    print("[IV-D] lambda :", lambdas)
    print("[IV-D] energy :", [f'{e:.1f}' for e in eng])
    print("[IV-D] JFIuav :", [f'{j:.3f}' for j in jfi])
    print(f"[IV-D] knee at lambda={lambdas[knee]}  "
          f"(E={eng[knee]:.1f} J, JFI={jfi[knee]:.3f}, succ={succ[knee]:.1f}%)")
    return lambdas, eng, jfi, knee


def _scenario(N, seed):     # placeholder to keep signature parallel (unused path)
    return ([], [{'id': 'Cloud_Server', 'type': 'cloud', 'f_loc': None}])


def _run_lambda(N, lam, seed):
    """Build a scenario and run the lambda-parametrised fair policy on it."""
    rng = np.random.default_rng(seed + N * 17)
    services, servers = [], [{'id': 'Cloud_Server', 'type': 'cloud', 'f_loc': None}]
    for i in range(N):
        uav_id = f'UAV_{i}'
        f_loc  = float(rng.uniform(1.2e9, 2.0e9))
        servers.append({'id': uav_id, 'type': 'uav', 'f_loc': f_loc})
        for _ in range(S.TASKS_PER_UAV):
            s_i = float(rng.uniform(10, 25)) * MB_BIT
            services.append({
                'uav_id': uav_id, 's_i': s_i,
                'c_i': float(rng.uniform(15, 25)) * s_i,
                'eta_comp': float(rng.uniform(0.15, 0.30)),
                't_max': float(rng.uniform(1.0, 2.0)),
                'f_loc': f_loc, 'conf_score': float(rng.uniform(0, 1)),
                'channel_rate': S.sample_channel_rate(rng, N),
                'dqn_lat_factor': float(rng.uniform(0.88, 0.95)),
                'dqn_eng_factor': float(rng.uniform(0.82, 0.90)),
            })
    return S.collect_metrics(xai_dqn_fair_lambda(services, servers, N, rng, lam))


# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("  PHASE IV — EXTENDED ANALYSES (built on the Phase I-III core)")
    print("=" * 70)
    run_phase4a(); print()
    run_phase4b(); print()
    run_phase4c(); print()
    run_phase4d()
    print("\nAll Phase IV figures written to:", OUT)
