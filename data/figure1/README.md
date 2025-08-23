# Figure 1: Synthetic Data Analysis

This directory contains scripts and figures for Figure 1 of the paper.

## Figure 1A: Income and Growth Rate Time Series

**File:** `figure1A.py`

**Purpose:** Demonstrates the time evolution of incomes and growth rates in the synthetic agent data.

**What it shows:**
- **Top subplot**: Time series of log incomes for all agents
  - X-axis: Time step
  - Y-axis: Log income (log dollars)
  - Each line represents one agent's trajectory
  - Mean and median trajectories highlighted
- **Bottom subplot**: Time series of growth rates for all agents
  - X-axis: Time step  
  - Y-axis: Growth rate (fractional change)
  - Each line represents one agent's growth rate trajectory
  - Mean and median trajectories highlighted

**Key insights:**
- Visualizes the income dynamics over time in the synthetic environment
- Shows how growth rates evolve over time for different agents
- Demonstrates the Kelly betting framework's temporal income evolution
- Provides baseline for comparing with real-world data
- Highlights population-level trends through mean/median trajectories

**Output files:**
- `figure1A_time_series.png` - High-resolution PNG
- `figure1A_time_series.pdf` - Publication-ready PDF

**Usage:**
```bash
cd figure1
python figure1A.py
```

**Requirements:**
- Dummy data must be generated first using `generate_dummy_data.py`
- Data file: `dummy_data_kelly_betting_dynamic_x.pkl` 