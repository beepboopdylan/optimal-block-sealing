"""
Generate a sample mempool CSV for testing the app's "Upload CSV" feature.

Simulates one Ethereum slot worth of pending transactions with realistic
priority fee tips and gas limits. Output matches the format expected by
the app (value, gas, arrival_time columns).

Usage:
    python generate_sample_data.py
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 300  # transactions pending in the mempool over one 12-second slot

# Priority fee tip per gas unit (Gwei) — heavy-tailed like real mempools
# Most transactions tip 1-3 Gwei, a few tip 10-50+ Gwei
base_tips = rng.exponential(scale=2.0, size=n)
spike_tips = rng.pareto(1.5, n) * 0.5
tips_gwei = base_tips + spike_tips
tips_gwei = np.clip(tips_gwei, 0.01, 100.0)

# Gas limits — common Ethereum transaction types
gas_types = rng.choice(
    [21_000, 46_000, 65_000, 120_000, 200_000, 300_000],
    p=[0.30, 0.15, 0.25, 0.15, 0.10, 0.05],
    size=n,
)
gas_limits = (gas_types + rng.integers(-2000, 2001, n)).clip(21_000, 500_000)

# Value to the builder = tip * gas (total priority fee earned, in Gwei)
# Scaled down so numbers are in a similar range to the simulation
values = (tips_gwei * gas_limits / 1e6).round(6)

# Arrival times: uniformly spread across a 12-second slot
arrival_times = np.sort(rng.uniform(0, 12.0, n)).round(3)

df = pd.DataFrame({
    "value":        values,
    "gas":          gas_limits.astype(int),
    "arrival_time": arrival_times,
})

out = "sample_mempool.csv"
df.to_csv(out, index=False)

print(f"Wrote {out}  ({n} transactions)")
print()
print(df[["value", "gas"]].describe().round(4))
