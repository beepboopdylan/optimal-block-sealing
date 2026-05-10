"""Block Sealing Strategy Tester — Streamlit interface."""

import streamlit as st
import numpy as np

from simulation import (
    generate_stream, run_algorithm, two_builder_race,
    offline_knapsack, compute_threshold, ALGORITHMS,
)
from figures import (
    value_comparison_chart,
    value_history_chart,
    threshold_evolution_chart,
    cstar_chart,
    race_chart,
    win_rate_chart,
    value_distribution_chart,
    flashbots_distribution_chart,
    competitive_ratio_table,
)
from data_sources import fetch_flashbots_bids, bids_to_transactions, parse_csv_upload

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Block Sealing Strategy Tester",
    layout="wide",
)

st.title("Block Sealing Strategy Tester")
st.caption(
    "Compare optimal stopping algorithms for Ethereum block builders — "
    "how much value does each policy leave on the table?"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    # ── 1. Data source ────────────────────────────────────────────────────────
    st.subheader("1. Data Source")
    data_source = st.radio(
        "source",
        ["Synthetic", "Upload CSV", "Flashbots Relay"],
        label_visibility="collapsed",
    )

    # Defaults matching notebook conventions (LAM=15, MU=1, B=50, Exp(1) values)
    lam: float = 15.0
    mu: float = 1.0
    T: float = 1.0
    B: int = 50
    value_dist: str = "exponential"
    value_mean: float = 1.0
    gas_mean: int = 3
    seed: int = 42
    values = gas_costs = arrival_times = None
    flashbots_df = None
    flashbots_meta: dict = {}
    _data_error: str | None = None

    if data_source == "Synthetic":
        lam = float(st.slider("Arrival rate λ (tx/sec)", 1, 50, 15))
        mu = float(st.slider("Slot-end rate μ", 0.1, 5.0, 1.0, step=0.1))
        T = 1.0 / mu   # mean slot duration
        B = int(st.slider("Block capacity B (gas units)", 10, 200, 50, step=5))
        value_dist = st.selectbox(
            "Value distribution", ["exponential", "pareto", "lognormal"])
        value_mean = float(st.slider("Mean transaction value", 0.1, 5.0, 1.0, step=0.1))
        gas_mean = int(st.slider("Mean gas cost (1–5 range)", 1, 5, 3))
        seed = int(st.number_input("Random seed", value=42, step=1,
                                    min_value=0, max_value=9999))

    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        st.caption("Required columns: `value`, `gas`\nOptional: `arrival_time`")
        B = int(st.slider("Block capacity B", 10, 200, 50, step=5))
        lam = float(st.slider("Arrival rate λ (tx/sec)", 1, 50, 15))
        mu = float(st.slider("Slot-end rate μ", 0.1, 5.0, 1.0, step=0.1))
        T = 1.0 / mu
        if uploaded_file is not None:
            try:
                values, gas_costs, arrival_times = parse_csv_upload(uploaded_file, B)
            except ValueError as e:
                _data_error = str(e)

    elif data_source == "Flashbots Relay":
        n_slots_fetch = int(st.slider("Slots to fetch", 50, 500, 100))
        B = int(st.slider("Block capacity B", 10, 200, 50, step=5))
        st.info("Fetches from boost-relay.flashbots.net — λ and μ are fit from data.")
        mu = 1.0 / 12.0
        T = 12.0

    # ── 2. Algorithms ─────────────────────────────────────────────────────────
    st.subheader("2. Single Builder")
    selected_algos = st.multiselect(
        "Compare algorithms",
        ALGORITHMS,
        default=["Greedy (online)", "Thompson Sampling",
                 "Optimal Threshold (c*)", "Optimal (offline)"],
    )

    ts_lam_prior = lam
    if "Thompson Sampling" in selected_algos or "Optimal Threshold (c*)" in selected_algos:
        with st.expander("Thompson Sampling / c* parameters"):
            ts_lam_prior = float(
                st.slider("Prior belief on λ", 0.1, 50.0, float(lam), step=0.5,
                          key="ts_lam"))
            if data_source != "Synthetic":
                mu = float(st.slider("Slot-end rate μ", 0.1, 5.0, mu,
                                      step=0.1, key="ts_mu"))
                T = 1.0 / mu

        if "Optimal Threshold (c*)" in selected_algos:
            st.caption(
                "c\\*(b) is computed assuming Exp(1) values and gas ~ Uniform[1,5]. "
                "This matches the notebook's calibration."
            )

    # ── 3. Two-builder race ───────────────────────────────────────────────────
    st.subheader("3. Two-Builder Race")
    policy_a = st.selectbox(
        "Builder A policy",
        ["Optimal Threshold (c*)", "Thompson Sampling",
         "Greedy (online)", "Optimal (offline)"],
        index=0,
    )
    policy_b = st.selectbox(
        "Builder B policy",
        ["Greedy (online)", "Thompson Sampling",
         "Optimal Threshold (c*)", "Optimal (offline)"],
        index=0,
    )
    n_trials = int(st.slider("Number of trials", 50, 2000, 500, step=50))

    if data_source != "Synthetic":
        with st.expander("Race simulation parameters"):
            st.caption("Race always generates synthetic data.")
            lam = float(st.slider("Race λ", 1, 50, 15, key="race_lam"))
            mu = float(st.slider("Race μ", 0.1, 5.0, 1.0, step=0.1, key="race_mu"))
            T = 1.0 / mu
            value_dist = st.selectbox(
                "Value distribution", ["exponential", "pareto", "lognormal"],
                key="race_dist")
            value_mean = float(
                st.slider("Mean tx value", 0.1, 5.0, 1.0, step=0.1, key="race_vm"))
            gas_mean = int(st.slider("Mean gas cost", 1, 5, 3, key="race_gas"))

    st.divider()
    run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_single, tab_race = st.tabs(["Single Builder Analysis", "Two-Builder Race"])


def _placeholder(tab, msg="Configure parameters in the sidebar and click **Run Simulation**."):
    with tab:
        st.info(msg)


if not run_btn:
    _placeholder(tab_single)
    _placeholder(tab_race)
    st.stop()

# ── Load / generate data ──────────────────────────────────────────────────────

if _data_error:
    st.error(f"Data error: {_data_error}")
    st.stop()

if data_source == "Synthetic":
    values, gas_costs, arrival_times = generate_stream(
        lam, T, value_dist, value_mean, gas_mean, seed=seed)

elif data_source == "Upload CSV":
    if values is None:
        st.warning("Please upload a CSV file first.")
        _placeholder(tab_single, "Upload a CSV file to continue.")
        _placeholder(tab_race)
        st.stop()

elif data_source == "Flashbots Relay":
    with st.spinner("Fetching Flashbots relay data..."):
        flashbots_df, err = fetch_flashbots_bids(n_slots_fetch)
    if err or flashbots_df is None:
        st.error(f"Could not fetch Flashbots data: {err}")
        st.stop()
    values, gas_costs, arrival_times, flashbots_meta = bids_to_transactions(
        flashbots_df, B)
    if values is None:
        st.error("No usable bid data in relay response.")
        st.stop()
    lam = flashbots_meta['lam_fit']
    mu = flashbots_meta['mu_fit']

if len(values) == 0:
    st.warning("No transactions generated. Try increasing λ or T.")
    st.stop()

# ── Precompute c* once (used by both tabs) ────────────────────────────────────

needs_cstar = ('Optimal Threshold (c*)' in selected_algos or
               policy_a == 'Optimal Threshold (c*)' or
               policy_b == 'Optimal Threshold (c*)')

precomputed_cstar = None
if needs_cstar:
    with st.spinner("Computing optimal threshold c\\*(b)…"):
        precomputed_cstar = compute_threshold(int(B), ts_lam_prior, mu)

# ── Run single-builder algorithms ─────────────────────────────────────────────

if selected_algos:
    with st.spinner("Running algorithms…"):
        results = [
            run_algorithm(name, values, gas_costs, B, mu=mu,
                          lam_prior=ts_lam_prior, c_star=precomputed_cstar)
            for name in selected_algos
        ]
    offline_val = next(
        (r['total_value'] for r in results if r['name'] == 'Optimal (offline)'),
        offline_knapsack(values, gas_costs, int(B)),
    )
else:
    results = []
    offline_val = offline_knapsack(values, gas_costs, int(B))

# ── Run two-builder race ──────────────────────────────────────────────────────

with st.spinner(f"Running {n_trials} race trials…"):
    race = two_builder_race(
        policy_a, policy_b,
        lam=lam, T=T, B=B, mu=mu,
        value_dist=value_dist, value_mean=value_mean,
        gas_mean=gas_mean, n_trials=n_trials,
    )

# ── Single builder tab ────────────────────────────────────────────────────────

with tab_single:
    if not results:
        st.info("Select at least one algorithm in the sidebar.")
        st.stop()

    best = max(results, key=lambda r: r['total_value'])
    ts_result = next(
        (r for r in results if r['name'] == 'Thompson Sampling'), None)
    cstar_result = next(
        (r for r in results if r['name'] == 'Optimal Threshold (c*)'), None)

    # Headline metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions in stream", len(values))
    col2.metric("Optimal (offline) ceiling", f"{offline_val:.4f}")
    col3.metric("Best algorithm", best['name'])
    col4.metric(
        "Best competitive ratio",
        f"{best['total_value'] / offline_val:.3f}" if offline_val > 0 else "—",
    )

    st.divider()

    # Value comparison + history charts
    c_left, c_right = st.columns(2)
    with c_left:
        st.plotly_chart(value_comparison_chart(results, offline_val),
                        use_container_width=True)
    with c_right:
        st.plotly_chart(value_history_chart(results, arrival_times),
                        use_container_width=True)

    # Thompson threshold evolution
    if ts_result and ts_result['thresholds']:
        st.plotly_chart(
            threshold_evolution_chart(
                ts_result['thresholds'],
                arrival_times[:len(ts_result['thresholds'])]
                if arrival_times is not None else None,
            ),
            use_container_width=True,
        )

    # c*(b) curve
    if precomputed_cstar is not None:
        st.plotly_chart(cstar_chart(precomputed_cstar), use_container_width=True)
        st.caption(
            "c\\*(b) is the theoretically optimal acceptance threshold when b gas units "
            "remain. It is non-increasing in b: be more selective when the block is empty, "
            "more permissive as it fills up."
        )

    # Competitive ratio table
    st.subheader("Competitive Ratio vs. Optimal")
    st.dataframe(
        competitive_ratio_table(results, offline_val),
        use_container_width=True,
        hide_index=True,
    )

    # Flashbots: real distribution overlay
    if data_source == "Flashbots Relay" and flashbots_meta.get('raw_values') is not None:
        st.subheader("Real Bid Values — Flashbots Relay")
        c_l, c_r = st.columns(2)
        with c_l:
            st.plotly_chart(
                flashbots_distribution_chart(
                    flashbots_meta['raw_values'],
                    flashbots_meta['exp_lam'],
                ),
                use_container_width=True,
            )
        with c_r:
            st.metric("Slots fetched", flashbots_meta['n_slots'])
            st.metric("Fitted λ (bids/sec)", f"{flashbots_meta['lam_fit']:.2f}")
            st.metric("Mean bid value",
                      f"{flashbots_meta['value_mean_gwei']:.2f} Gwei")
        with st.expander("Raw relay data sample"):
            st.dataframe(flashbots_df.head(20), use_container_width=True)

# ── Two-builder race tab ──────────────────────────────────────────────────────

with tab_race:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Builder A win rate",
              f"{race['a_win_rate']:.1%}",
              delta=race['policy_a'])
    c2.metric(f"Builder B win rate",
              f"{race['b_win_rate']:.1%}",
              delta=race['policy_b'])
    c3.metric("Builder A mean value",
              f"{race['a_mean_value']:.4f}",
              delta=f"±{race['a_std']:.4f}")
    c4.metric("Builder B mean value",
              f"{race['b_mean_value']:.4f}",
              delta=f"±{race['b_std']:.4f}")

    st.divider()

    c_left, c_right = st.columns(2)
    with c_left:
        st.plotly_chart(race_chart(race), use_container_width=True)
    with c_right:
        st.plotly_chart(win_rate_chart(race), use_container_width=True)

    st.plotly_chart(
        value_distribution_chart(
            race['all_a_values'], race['all_b_values'],
            f"Builder A ({race['policy_a']})", f"Builder B ({race['policy_b']})",
        ),
        use_container_width=True,
    )

    avg = (race['a_mean_value'] + race['b_mean_value']) / 2
    diff = abs(race['a_mean_value'] - race['b_mean_value'])
    pct = diff / avg * 100 if avg > 0 else 0.0
    winner_label = race['policy_a'] if race['a_win_rate'] >= 0.5 else race['policy_b']
    st.info(
        f"**Efficiency loss from competition:** "
        f"mean value difference = {diff:.4f} ({pct:.1f}% of average). "
        f"**{winner_label}** wins more than half the time."
    )
