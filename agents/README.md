# Agents Directory Structure

This directory contains the implementations of our AI agents for the Billiards project.

## 🚀 Final Agent

- **`new_agent.py`**
  - **Status**: ✅ **Optimal / Final Version**
  - **Description**: This is the best-performing agent (equivalent to **v5** in the report). It integrates all successful features:
    - **Micro**: Warm-Start CMA-ES Optimization.
    - **Meso**: Beam Search Sequence Planning (Lookahead).
    - **Macro**: Meta-Strategies (Suicide Break, Minimax Defense).
  - **Usage**: Use this file for final benchmarks and competitions.

## 📂 Iteration History (`agent_all_versions/`)

The `agent_all_versions/` subdirectory contains the historical milestones of our development process. These files are preserved for ablation studies and to demonstrate the evolutionary path.

| File | Version ID | Key Features & Role |
|------|------------|---------------------|
| `new_agent_v2.py` | **v2** | **Optimization Core**. Introduced CMA-ES and basic Position Scoring. <br>(Solved mechanical accuracy). |
| `new_agent_v3.py` | **v3** | **Strategic Refinement**. Added "Force Attack" defense and "Suicide Break". <br>(Improved win rate against basic opponents). |
| `new_agent_v4.py` | **v4** | **Failed Experiment**. Attempted complex physics simulations without geometric priors. <br>(Documented for "Design Retrospective" to show what didn't work). |
| `new_agent_v5.py` | **v5** | **Planning Breakthrough**. Introduced Beam Search ($D=2$). <br>(The basis for the final `new_agent.py`). |

## ℹ️ Note
For the final project submission or presentation, please run **`new_agent.py`**. The files in the subfolder are primarily for reference and experimental analysis.
