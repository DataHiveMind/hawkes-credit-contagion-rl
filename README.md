# Hawkes-Credit-Contagion-RL: Graph-Attention Hawkes Processes & RL for CDS-Bond Basis Arbitrage

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![kdb+/q 4.1](https://img.shields.io/badge/kdb%2B%2Fq-4.1-0052CC.svg)
![PyKX](https://img.shields.io/badge/PyKX-Zero--Copy_IPC-00B8D9.svg)
![PyTorch Geometric](https://img.shields.io/badge/PyG-Graph_Attention-3C2179.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Credit_Desk_API-009688.svg)
![MLflow](https://img.shields.io/badge/MLflow-Governance-0194E2.svg)

An institutional quantitative credit platform that models **supply-chain and cross-lender default contagion** using a **Continuous-Time Multivariate Graph-Attention Hawkes Process** coupled with a **Distributional-Overlay Proximal Policy Optimization (PPO)** capital allocator. Powered by a **kdb+/q v4.1 tick plant**, the system aligns sparse FINRA TRACE corporate bond prints with continuous DTCC CDS quotes via native window joins (`wj`) to monetize **CDS-Bond Basis dislocations** while dynamically hedging tail cascades using CDX/iTraxx tranches.

---

## 1. Enterprise Usage Across Internal Teams

| Internal Team | Interface | Operational Workflow |
| :--- | :--- | :--- |
| **Credit Stat-Arb & Fixed Income PMs** | `services/desk_ui/` (Contagion & Basis Portal) | Visualizes live mutually-exciting hazard intensities across corporate supply chains and screens rich/cheap Par-Equivalent CDS-Bond Basis dislocations. |
| **Counterparty & Enterprise Risk Officers** | `POST /v1/contagion/simulate-shock` | Injects hypothetical rating downgrades or spread jumps into a specific issuer node and simulates 5-to-30-day contagion cascades across the firm's credit book. |
| **Systematic Credit Execution Desk** | `GET /v1/basis/top-dislocations` & `eod_desk_blotter.py` | Pulls capital-constrained long/short basis notional weights and optimal CDX IG/HY tranche tail-hedge ratios directly into the execution blotter. |
| **Quantitative Credit Researchers** | `src/credit_hawkes_rl/` (Internal SDK) | Imports `credit_hawkes_rl` to access the PyKX-backed dynamic graph builder, Hawkes compensator integrators, and par-equivalent basis calculators. |

---

## 2. Mathematical Formulation

### 2.1 Multivariate Graph-Attention Neural Hawkes Process
Let $\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t, \mathbf{W}_t)$ represent a directed corporate credit network of $N = \vert{}\mathcal{V}\vert{}$ reference entities, where edges $(j \to i) \in \mathcal{E}_t$ encode supplier-customer revenue dependency, shared bank syndicate lending, and sector overlap. Given a sequence of credit jump events $\mathcal{H}_t = \{(t_k, u_k, m_k) : t_k < t\}$ (where $u_k \in \mathcal{V}$ is the issuer and $m_k$ is the spread jump magnitude or rating migration), the conditional hazard intensity $\lambda_i^*(t)$ for issuer $i$ is:

$$\lambda_i^*(t) = \text{softplus}\left( \mu_i + \sum_{(t_k, u_k, m_k) \in \mathcal{H}_t : u_k \in \mathcal{N}(i)} \alpha_{i, u_k}(t_k) \, \phi(m_k) \, \kappa_{i, u_k}(t - t_k) \right)$$

where $\alpha_{i, u_k}(t_k)$ is a dynamic **Graph Attention (GAT)** excitation weight learned over structural balance-sheet embeddings $\mathbf{h}_i(t_k)$, and $\kappa_{i,j}(\Delta t) = \sum_{b=1}^B w_{ij,b} \beta_b \exp(-\beta_b \Delta t)$ is a multi-scale exponential decay kernel maintained in real time inside `kdb_database/scripts/cep_hawkes_kernel.q`.

The model parameters are trained by maximizing the exact point-process log-likelihood over $[0, T]$:

$$\mathcal{L}_{\text{Hawkes}}(\Theta) = \sum_{k : t_k \le T} \log \lambda_{u_k}^*(t_k) - \sum_{i=1}^N \int_0^T \lambda_i^*(t) \, dt$$

### 2.2 Par-Equivalent CDS-Bond Basis & Net Carry
For issuer $i$ and tenor $\tau$, the **CDS-Bond Net Basis** $B_{i,t}(\tau)$ measures the dislocation between the synthetic credit spread in the derivative market ($\text{CDS}_{i,t}(\tau)$) and the cash bond Z-spread ($Z_{i,t}(\tau)$), adjusted for bond price deviation from par, repo financing hair-cuts $r_{\text{repo}, i}$, and cheapest-to-deliver (CTD) option value:

$$B_{i,t}(\tau) = \text{PE-CDS}_{i,t}(\tau) - Z_{i,t}(\tau)$$

* **Negative Basis Trade ($B_{i,t} < 0$):** Buy the cash corporate bond (financed via repo) + buy protection via single-name CDS to lock in risk-free positive carry while insulating against default.
* **Positive Basis Trade ($B_{i,t} > 0$):** Short the cash corporate bond (via reverse repo) + sell CDS protection.

### 2.3 Contagion-Aware PPO Capital Allocator with $\text{CVaR}_{\alpha}$ Overlay
The RL allocator outputs portfolio weights $\mathbf{w}_t \in \mathbb{R}^N$ across single-name basis pairs and tail-hedge weights $\mathbf{h}_t^{\text{CDX}}$ across CDX/iTraxx index tranches to maximize risk-adjusted carry-and-convergence PnL subject to a Jump-to-Default Conditional Value-at-Risk ($\text{CVaR}_{0.01}$) constraint derived from the Hawkes intensity vector $\boldsymbol{\lambda}^*(t)$:

$$\max_{\pi_\theta} \; \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^T \gamma^t \left( \mathbf{w}_t^\top \Delta \mathbf{B}_{t+1} + \mathbf{w}_t^\top \mathbf{Carry}_t + \left(\mathbf{h}_t^{\text{CDX}}\right)^\top \mathbf{r}_{t+1}^{\text{CDX}} - \text{TC}(\Delta \mathbf{w}_t) \right) \right] \quad \text{s.t.} \quad \text{CVaR}_{0.01}\left(\text{JTD}(\mathbf{w}_t, \boldsymbol{\lambda}^*(t))\right) \le \bar{L}_{\text{desk}}$$

---

## 3. kdb+/q v4.1 & PyKX Tick-Plant Architecture

In corporate credit, FINRA TRACE bond executions occur asynchronously relative to streaming dealer CDS quotes. Our kdb+ CEP engine (`cep_cds_bond_basis.q`) performs vectorized **window joins (`wj`)** and **as-of joins (`aj`)** in memory:

```q
/ Vectorized window join pairing every FINRA TRACE bond print with +/- 30s dealer CDS spread quotes
w: -30000000000 0 +\: trace_bond_print`time;
basis_snap: wj[w; `ticker`time; trace_bond_print; (cds_quote; (avg; `par_spread); (last; `upfront_pts))];
```

* **`tick.q (:5010)`:** Ingests FINRA TRACE prints, DTCC CDS ticks, repo rates, and rating agency actions.
* **`r.q (:5011)` & `hdb/ (:5012)`:** Maintains intraday credit curves in RAM and splays date-partitioned tables (`#p` parted on `ticker`) at EOD.
* **`cep_hawkes_kernel.q (:5013)`:** Updates recursive exponential decay accumulators $A_{ij,b}(t) = \exp(-\beta_b \Delta t) A_{ij,b}(t^-) + \phi(m)$ on every spread jump.
* **`pykx_graph_builder.py`:** Maps kdb+ node intensity tables and edge lists directly into **PyTorch Geometric (`HeteroData`)** tensors via zero-copy NumPy buffers.

---

## 4. Repository Structure

```text
hawkes-credit-contagion-rl/
├── .devcontainer/                 # VS Code Dev Container (Python 3.12, PyKX, PyG, kdb+ 4.1)
├── .vscode/                       # Launch configurations for Hawkes GAT trainer & Credit API
├── configs/                       # Hydra configs (supply-chain topology, Hawkes kernels, PPO limits)
├── data/                          # Schemas for TRACE bond prints, DTCC CDS, RED codes, and ISINs
├── kdb_database/                  # Institutional Credit kdb+/q Tick Plant
│   ├── schema.q                   # Tables: trace_bond_print, cds_quote, supply_chain_edge, basis_signal
│   └── scripts/                   # cep_cds_bond_basis.q (wj/aj) & cep_hawkes_kernel.q
├── src/credit_hawkes_rl/          # Installable internal Python SDK
│   ├── kdb_bridge/                # PyKX Gateway client & zero-copy PyG HeteroData tensor builder
│   ├── graph_builder/             # Supply-chain DAG and shared bank syndicate edge constructors
│   ├── models/                    # Multivariate Graph-Attention Neural Hawkes & compensator solver
│   ├── basis_engine/              # Z-spread, Par-Equivalent CDS spread, and repo carry engine
│   ├── rl_allocator/              # Contagion Gymnasium env, PPO allocator, and CVaR JTD overlay
│   └── reporting/                 # Writes trade blotters back to kdb+ `desk_blotter` table
├── services/
│   ├── api/                       # FastAPI service (/v1/contagion/simulate-shock, /v1/basis/top-dislocations)
│   └── desk_ui/                   # Interactive network contagion map & basis screener for PMs
├── mlops/                         # MLflow tracking & Model Risk Management (MRM) audit logs
├── tests/                         # Unit tests verifying q window joins and Hawkes log-likelihood
├── docker/                        # Production Dockerfiles & docker-compose orchestration
├── Makefile                       # Standardized operational CLI commands
├── environment.yml                # Conda environment specification
└── pyproject.toml                 # Package metadata & CLI entrypoint (`credit-engine`)
```

---

## 5. Quickstart & Production Operations

### 5.1 Setup & Initialization
```bash
conda env create -f environment.yml
conda activate credit_hawkes_env
pip install -e ".[dev]"
export QLIC=/opt/kx/lic
```

### 5.2 Running the Credit Tick Plant, Hawkes Calibration & Desk Portal
```bash
# 1. Start kdb+ Tickerplant, RDB, HDB, Hawkes CEP Engine, and Gateway
make kdb-up

# 2. Calibrate the Graph-Attention Neural Hawkes model & train the PPO Basis Allocator
credit-engine train hawkes_gat=multiscale_3b ppo_allocator.cvar_limit_bps=45.0

# 3. Launch the Credit Desk API (:8001) and Contagion Web Portal (:8502)
make serve-all
```

### 5.3 Example FastAPI Request (Enterprise Risk Shock Simulation)
```bash
curl -X POST "http://localhost:8001/v1/contagion/simulate-shock" \
  -H "Content-Type: application/json" \
  -d '{
    "shocked_issuer_ticker": "F",
    "spread_jump_bps": 175.0,
    "horizon_days": 10,
    "confidence_level": 0.99
  }'
```