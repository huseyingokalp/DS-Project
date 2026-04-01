# Energy-Aware Hybrid Edge-Cloud Computing Architecture for Real-Time Plant Health Analysis in Resource-Constrained UAV Networks 

> **11117BLG016 Distributed Systems** — Süleyman Demirel University  
> **Student:** Hüseyin GÖKALP · d2540138002@ogr.sdu.edu.tr  
> **Instructor:** Assoc. Prof. Dr. Asım Sinan YÜKSEL  
> **Status:** 🟡 Week 7 Mid-Term Report Submitted — Phase II/III ongoing toward Final

---

## Overview

UAV fleets deployed in precision agriculture generate high-resolution multispectral images at scale. Transmitting raw frames to a centralized cloud introduces severe network latency; processing them entirely on-board depletes flight batteries. This project resolves both bottlenecks simultaneously by:

- Repurposing **Grad-CAM (XAI)** as an active pre-transmission filter — healthy pixels are spatially zeroed, reducing payload size to 15–30% of the original,
- Deploying a **Deep Q-Network (DQN)** agent that autonomously decides, per task, whether to process locally at the edge or offload to the cloud.

---

## Repository Structure

```
DS-Project/
├── Analyse/
│   ├── Phase I/
│   │   ├── Comparative_Analysis.ipynb              # Phase I deterministic baseline simulation
│   │   ├── results_latency.png                     # Phase I latency results
│   │   ├── results_energy.png                      # Phase I energy results
│   │   └── results_success.png                     # Phase I success results
│   ├── Phase II-III/
│   │   ├── Improved_Comparative_Analysis_v1.ipynb  # Phase II/III stochastic simulations (ongoing)
│   │   ├── fig1_latency_revised.png                # Avg. end-to-end latency (Phase II/III)
│   │   ├── fig2_energy_revised.png                 # Avg. UAV energy consumption (Phase II/III)
│   │   ├── fig3_success_revised.png                # Task success ratio
│   │   ├── fig4_bar_summary_N50.png                # N=50 summary comparison
│   │   ├── fig5_radar_chart_N50.png                # Multi-dimensional radar chart
│   │   ├── fig6a_efficiency.png                    # Energy efficiency (EE)
│   │   ├── fig6b_offload.png                       # Cloud offload ratio
│   │   ├── fig6c_fairness.png                      # Jain's Fairness Index (JFI)
│   │   └── fig7_improvement_heatmap.png            # Improvement heatmap
├── Diagram/
│   ├── Fig_System_Architecture.mmd             # Three-tier architecture (Mermaid source)
│   ├── Fig_System_Architecture.png             # System architecture figure
│   ├── poc_gradcam.mmd                         # Grad-CAM PoC block diagram (Mermaid source)
│   └── poc_gradcam.png                         # Grad-CAM PoC figure
└── References/
    └── references.bib                          # IEEE-formatted bibliography
```

---

## Methodology

### System Architecture (3-Tier)

```
[Tier 1 — Ground]   Soil sensors + Agricultural crops
        ↓
[Tier 2 — Edge]     UAV fleet → Lightweight CNN → Grad-CAM filter → Compressed ROI payload
        ↓  (15–30% of original size)
[Tier 3 — Cloud]    Centralized server → Deep analysis → Feedback
```

### Core Equations

**XAI Data Compression:**
```
s'_i = s_i × η_comp,   η_comp ∈ [0.15, 0.30]
```

**Channel Rate (Rayleigh fading + co-channel interference):**
```
R_i = B · log2(1 + P_i·h_i / (σ² + I_co))
```

**DQN Reward Function:**
```
R_t = -(ω₁·T + ω₂·E) - ρ·P_drop
```
> ⚠️ Final version will add a JFI penalty term: `- ω₃·(1 - JFI)`

---

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Bandwidth | 20 MHz |
| UAV count (N) | 10–50 |
| Raw image size | 10–25 MB/frame |
| XAI compression ratio (η_comp) | 15%–30% |
| Hovering power | 120 W |
| Monte Carlo seeds (Phase II/III) | M = 5 |
| CPU contention factor | φ_max = 1.35 |

---

## Results (As of Week 7)

### Phase I — Deterministic Baseline (Completed)

| Algorithm | Latency (s) | Energy (J) | Success (%) |
|-----------|-------------|------------|-------------|
| Cloud-Only | 3.99 | 479.1 | 13.2 |
| Edge-Only | 1.83 | 219.6 | 35.0 |
| XAI-Heuristic | 0.56 | 67.2 | **98.6** |
| XAI-DQN (Dry-Run) | **0.52** | **58.0** | 92.8 |

*Measured at N=50 UAVs under deterministic conditions.*

### Phase II/III — Preliminary Stochastic Results (Ongoing)

- DQN confines average latency to **0.491 s** under multi-seed Rayleigh fading
- Achieves **14.0% energy reduction** over the XAI-Heuristic baseline
- **Critical issue:** Jain's Fairness Index drops to **≈0.40** at N=50 — the DQN's utilitarian policy overloads a subset of UAVs while optimizing aggregate energy, creating potential aerial dead zones

---

## Tools & Frameworks

| Tool | Purpose |
|------|---------|
| [EdgeSimPy](https://github.com/EdgeSimPy/EdgeSimPy) | Python-native edge computing simulator |
| [EdgeAISim](https://github.com/nandhakumar-rs/EdgeAISim) | EdgeSimPy extension with built-in DRL support |
| Grad-CAM (PyTorch) | XAI-driven spatial data compression |
| Deep Q-Network | Autonomous computation offloading policy |
| LaTeX / IEEEtran | Academic paper formatting |

---

## Roadmap

- [x] Phase I: XAI-Heuristic deterministic baseline
- [x] Phase I: Mathematical modeling (Grad-CAM, channel model, MDP formulation)
- [x] Phase II: Stochastic model formulation (Rayleigh fading + CPU contention)
- [x] Phase II: DQN preliminary training and dry-run

---

## Setup

```bash
git clone https://github.com/huseyingokalp/DS-Project.git
cd DS-Project

pip install edge-sim-py gymnasium torch numpy matplotlib
```

Run the notebooks in order:
1. `Analyse/Phase I/Comparative_Analysis.ipynb` → Phase I baseline
2. `Analyse/Phase II-III/Improved_Comparative_Analysis_v1.ipynb` → Phase II/III stochastic simulations
