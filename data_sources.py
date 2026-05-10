"""Data loading utilities: Flashbots relay API, CSV upload, synthetic generation."""

import requests
import numpy as np
import pandas as pd
import streamlit as st


FLASHBOTS_RELAY = "https://boost-relay.flashbots.net"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_flashbots_bids(n_slots: int = 100):
    """
    Fetch recent delivered payload bids from the Flashbots relay.

    Returns (DataFrame, error_string). On success error_string is None.
    Each row is one delivered block with columns: slot, value_gwei, gas_used, gas_limit.
    """
    url = f"{FLASHBOTS_RELAY}/relay/v1/data/bidtraces/proposer_payload_delivered"
    try:
        resp = requests.get(url, params={"limit": n_slots}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None, "Relay returned empty response."
        df = pd.DataFrame(data)
        df['value_gwei'] = df['value'].astype(float) / 1e9
        df['gas_used'] = df['gas_used'].astype(int)
        df['gas_limit'] = df['gas_limit'].astype(int)
        df['slot'] = df['slot'].astype(int)
        df = df.sort_values('slot', ascending=False).reset_index(drop=True)
        return df, None
    except requests.exceptions.Timeout:
        return None, "Request timed out (relay may be slow). Try again."
    except Exception as e:
        return None, str(e)


def bids_to_transactions(df: pd.DataFrame, B: int = 50, seed: int = 42):
    """
    Convert relay bid data into a synthetic transaction stream.

    Fits λ from the number of bids fetched and uses μ = 1/12 (Ethereum slot = 12 s).
    Returns (values, gas_costs, arrival_times, meta_dict).
    """
    raw_values = df['value_gwei'].values
    raw_values = raw_values[raw_values > 0]
    if len(raw_values) == 0:
        return None, None, None, {}

    rng = np.random.default_rng(seed)
    n = len(raw_values)

    # Normalise values so their mean matches exponential(1) scale for B
    target_mean = 1.0
    scale = target_mean / raw_values.mean()
    values = raw_values * scale

    # Gas costs: uniform integers in [1, 5] to match compute_threshold's assumption
    gas_costs = rng.integers(1, 6, n).astype(float)

    # Spread arrivals across a 12-second slot
    arrival_times = np.sort(rng.uniform(0, 12.0, n))

    lam_fit = n / 12.0      # bids per second
    mu_fit = 1.0 / 12.0     # Ethereum slot is always ~12 s
    exp_lam = 1.0 / raw_values.mean() if raw_values.mean() > 0 else 1.0

    meta = {
        'lam_fit': lam_fit,
        'mu_fit': mu_fit,
        'exp_lam': exp_lam,
        'value_mean_gwei': float(raw_values.mean()),
        'value_std_gwei': float(raw_values.std()),
        'n_slots': len(df),
        'raw_values': raw_values,
    }
    return values, gas_costs, arrival_times, meta


def parse_csv_upload(file, B: int = 50):
    """
    Parse an uploaded CSV with transaction data.

    Required columns: value, gas  (case-insensitive, accepts common aliases)
    Optional columns: arrival_time / timestamp / t

    Gas costs are normalised to [1, 5] if they arrive in raw Ethereum units,
    keeping them consistent with the c*(b) computation assumptions.
    Returns (values, gas_costs, arrival_times).
    """
    df = pd.read_csv(file)
    df.columns = [c.lower().strip() for c in df.columns]

    value_col = next(
        (c for c in ['value', 'tip', 'fee', 'reward', 'val'] if c in df.columns), None)
    gas_col = next(
        (c for c in ['gas', 'gas_cost', 'gas_used', 'gas_limit', 'cost'] if c in df.columns),
        None)
    time_col = next(
        (c for c in ['arrival_time', 'time', 'timestamp', 't'] if c in df.columns), None)

    if value_col is None:
        raise ValueError(
            "No value column found. Expected one of: value, tip, fee, reward.")
    if gas_col is None:
        raise ValueError(
            "No gas column found. Expected one of: gas, gas_cost, gas_used, gas_limit.")

    values = df[value_col].astype(float).values
    gas_costs = df[gas_col].astype(float).values

    # Normalise gas costs to [1, 5] range to match c*(b) computation assumptions
    if gas_costs.max() > 5:
        gas_costs = np.clip(
            np.round(gas_costs / gas_costs.max() * 5).astype(int), 1, 5
        ).astype(float)

    arrival_times = (df[time_col].astype(float).values if time_col
                     else np.linspace(0, 12.0, len(values)))

    return values, gas_costs, arrival_times
