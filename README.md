# Energy-Aware Hybrid Edge-Cloud Computing for Real-Time Plant Health Analysis in UAV Networks

> Purposing **Grad-CAM (XAI)** as a *pre-transmission data-compression* mechanism to cut UAV offloading energy, evaluated through a four-phase progressive framework on **EdgeSimPy + EdgeAISim**.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Simulator](https://img.shields.io/badge/Simulator-EdgeSimPy%20%2B%20EdgeAISim-green.svg)](https://github.com/EdgeSimPy/EdgeSimPy)
[![Dataset](https://img.shields.io/badge/Dataset-PlantVillage-orange.svg)](https://www.tensorflow.org/datasets/catalog/plant_village)
[![Made with](https://img.shields.io/badge/Made%20with-Jupyter%20%2F%20Colab-F37626.svg)](https://colab.research.google.com/)

-----

## Overview

UAV-assisted precision agriculture must diagnose plant diseases in real time, yet transmitting raw multispectral imagery (~25 MB/frame) to the cloud is slow and energy-hungry, while computing everything on-board drains the battery. This project proposes a **hybrid edge-cloud architecture** in which the UAV first runs a lightweight CNN and a **Grad-CAM explainability layer that doubles as a compression filter**: healthy-tissue and background pixels are zeroed out, so only the pathogenic Regions of Interest (ROI) — roughly **15–30 % of the original payload** — are offloaded. A reinforcement-learning agent then decides, per task, whether to process locally or offload.

The central hypothesis: the XAI-supported hybrid architecture reduces average UAV flight energy by **> 85 %** versus a cloud-only approach while holding **sub-second latency**, and the adaptive DRL agent yields a further **≥ 14 %** energy reduction over a non-adaptive heuristic.

This repository contains all simulation code, figures, diagrams, and the reference bibliography behind the accompanying IEEE-format manuscript.

-----

## The Four-Phase Progressive Framework

The contribution is **not** a monotonically rising accuracy score, but the progressive closure of four *distinct* deficiencies. Each phase optimises a different objective, so metrics are **not directly comparable across phases**.

|Phase                             |Focus                        |What it adds                                                                  |Key outcome                                                         |
|----------------------------------|-----------------------------|------------------------------------------------------------------------------|--------------------------------------------------------------------|
|**I — Deterministic Baseline**    |Latency / energy feasibility |Cloud-Only, Edge-Only, XAI-Heuristic, XAI-DQN (dry-run) under ideal conditions|Establishes that XAI ROI compression is the decisive mechanism      |
|**II — Stochastic Robustness**    |Reliability under uncertainty|Rayleigh fading + dynamic CPU contention, multi-seed Monte Carlo, trained DQN |DQN *sustains* ~92.8 % success and best energy efficiency under load|
|**III — Fairness-Aware Extension**|Equity across the fleet      |Jain-shaped reward + load-augmented state → **XAI-DQN-Fair**                  |Per-UAV Jain index restored **0.789 → 0.952** at marginal cost      |
|**IV — Assumption Validation**    |Real-world validity          |Grad-CAM overhead, η sweep, in-field noise, λ-Pareto front                    |Validates the model and candidly reports its limits/regressions     |

**Compared algorithms:** `Cloud-Only` · `Edge-Only` · `XAI-Heuristic` · `XAI-DQN` · `XAI-DQN-Fair`

-----

## Key Results (N = 50 UAVs)

### Phase I — Deterministic baseline

|Algorithm          |Latency (s)|Energy (J)|Success (%)|
|-------------------|:---------:|:--------:|:---------:|
|Cloud-Only         |3.99       |479.1     |13.2       |
|Edge-Only          |1.83       |219.6     |35.0       |
|XAI-Heuristic      |0.56       |67.2      |98.6       |
|XAI-DQN *(dry-run)*|0.52       |58.0      |92.8       |

### Phase II / III — Stochastic + fairness (multi-seed, Rayleigh fading)

|Algorithm       |Latency (s)|Energy (J)|Success (%)|JFI (per-UAV)|
|----------------|:---------:|:--------:|:---------:|:-----------:|
|Cloud-Only      |3.573      |429.0     |12.0       |0.642        |
|Edge-Only       |2.412      |289.8     |12.8       |0.972        |
|XAI-Heuristic   |0.538      |64.6      |98.4       |0.787        |
|**XAI-DQN**     |**0.491**  |**55.5**  |92.8       |0.789        |
|**XAI-DQN-Fair**|0.522      |59.1      |92.5       |**0.952**    |

*XAI-DQN energy efficiency ≈ 16.82 × 10³ Tasks/J · offload ratio ≈ 74.2 %.*

### Phase IV — Validation highlights

- **IV-A (Grad-CAM overhead):** ~**5.33 J/task** on-UAV cost; net energy advantage holds beyond a **~1.5 MB** break-even.
- **IV-B (η sensitivity):** graceful degradation up to η = 0.90, with a strict-real-time inflection at **η* ≈ 0.65**.
- **IV-C (in-field noise):** lab-to-field success drop bounded to **≈ 8–11 pp**.
- **IV-D (λ-Pareto):** energy–fairness frontier with a **win–win knee at λ ≈ 0.1**, dominating the hand-picked Phase III value (λ = 0.5).

> **Overall recommended configuration:** `XAI-DQN-Fair` operated at **λ ≈ 0.1**.

-----

## Repository Structure

```
DS-Project/
├── Analyse/
│   ├── Phase I/
│   │   └── Comparative_Analysis.ipynb            # Deterministic baseline (Table III + Figs)
│   ├── Phase II-III/
│   │   └── Improved_Comparative_Analysis_v3.ipynb # Stochastic + fairness (Table IV + Figs)
│   └── Phase IV/
│       └── Phase4_Extended_Analysis_Colab.ipynb  # 4 validation analyses (Figs 16–19)
├── Diagram/
│   ├── Fig_System_Architecture.png / .mmd        # Three-tier hybrid edge-cloud architecture
│   └── poc_gradcam.png / .mmd                     # Grad-CAM compression proof-of-concept
└── References/
    └── references.bib                            # Bibliography (IEEE)
```

-----

## Getting Started

### Run in Google Colab (recommended)

Each notebook is self-contained and reproduces the exact paper figures and numbers. Open any notebook in Colab and run all cells:

- **Phase I:** `Analyse/Phase I/Comparative_Analysis.ipynb`
- **Phase II & III:** `Analyse/Phase II-III/Improved_Comparative_Analysis_v3.ipynb`
- **Phase IV:** `Analyse/Phase IV/Phase4_Extended_Analysis_Colab.ipynb`

### Run locally

```bash
git clone https://github.com/huseyingokalp/DS-Project.git
cd DS-Project

# Core dependencies (Phases II, III, IV)
pip install numpy scipy matplotlib

# Phase I additionally builds an explicit topology with EdgeSimPy
pip install edge-sim-py networkx

# Launch the notebooks
jupyter notebook
```

**Suggested run order:** Phase I → Phase II/III → Phase IV.

-----

## Simulation Setup

- **Simulators:** EdgeSimPy (discrete-event edge/cloud modelling) + EdgeAISim (DRL/AI toolkit)
- **Dataset:** [PlantVillage](https://www.tensorflow.org/datasets/catalog/plant_village) — ~54,000 images across 38 crop-disease classes; assumed baseline diagnostic accuracy 97.7–98.0 % (cf. EfficientNet/MobileNetV3 benchmarks)
- **DQN action space:** offload decision `xᵢ ∈ {0,1}` × XAI threshold `τ_XAI ∈ {0.15, 0.20, 0.25, 0.30}`

### Core parameters

|Symbol   |Meaning                        |Value      |
|---------|-------------------------------|-----------|
|`B`      |Bandwidth                      |20 MHz     |
|`Pᵢ`     |Transmission power             |20 dBm     |
|`σ²`     |Noise floor                    |−100 dBm   |
|`γ`      |Co-channel coupling            |0.00035    |
|`κ`      |Switched capacitance           |10⁻²⁸      |
|`P_hover`|Hovering power                 |120 W      |
|`f_loc`  |Edge CPU frequency             |1.2–2.0 GHz|
|`ω₁, ω₂` |Latency/energy weights         |0.5, 0.5   |
|`η_comp` |XAI compression ratio          |15–30 %    |
|`M`      |Monte-Carlo seeds (Phase II–IV)|5          |
|`λ`      |Fairness weight (Phase III)    |0.5        |
|`E_cap`  |Catastrophic energy cap        |250 J      |

### Phase IV-specific

`ζ = 2.4` (Grad-CAM cost factor) · `P_SBC = 3.4 W` · `t_fwd = 0.018 s` · η sweep 0.15–0.90 · `T_strict = 1.0 s` · in-field noise `ν ∈ [0,1]` · λ-grid {0, 0.1, 0.25, 0.5, 0.75, 1.0}.

-----

## System Architecture

The architecture spans three tiers — **IoT/Device** (crops, soil sensors) → **Edge** (UAV fleet running the lightweight DNN, Grad-CAM filter, spatial cropping, and the DRL offloading agent) → **Cloud** (centralised heavy computation and global aggregation). See `Diagram/Fig_System_Architecture.png` and the Grad-CAM compression proof-of-concept in `Diagram/poc_gradcam.png`.

-----

## Course Information

- **Course:** 11117BLG016 — Distributed Systems (Dağıtık Sistemler)
- **Institution:** Süleyman Demirel University, Isparta, Türkiye
- **Instructor:** Assoc. Prof. Dr. Asım Sinan Yüksel
- **Author:** Hüseyin Gökalp

-----

## Citation

If you use this work, please cite the accompanying manuscript:

```bibtex
@misc{gokalp_energyaware_uav,
  author = {Hüseyin Gökalp},
  title  = {Energy-Aware Hybrid Edge-Cloud Computing Architecture for Real-Time
            Plant Health Analysis in Resource-Constrained UAV Networks},
  year   = {2025},
  note   = {Distributed Systems (11117BLG016), Süleyman Demirel University},
  howpublished = {\url{https://github.com/huseyingokalp/DS-Project}}
}
```
