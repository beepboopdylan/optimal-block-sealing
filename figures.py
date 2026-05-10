"""Plotly chart builders for the Block Sealing Strategy Tester."""

import plotly.graph_objects as go
import pandas as pd
import numpy as np

COLOR_MAP = {
    'Greedy (online)':         '#e74c3c',
    'Greedy (unsorted)':       '#e74c3c',
    'Thompson Sampling':       '#3498db',
    'Optimal Threshold (c*)':  '#9b59b6',
    'Optimal (offline)':       '#2ecc71',
    'Greedy (density-sorted)': '#e67e22',
    'Builder A':               '#3498db',
    'Builder B':               '#e74c3c',
}

_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(size=13),
    margin=dict(t=60, b=40, l=50, r=20),
)


def value_comparison_chart(results, offline_value=None):
    """Bar chart comparing total block value captured by each algorithm."""
    names = [r['name'] for r in results]
    values = [r['total_value'] for r in results]
    colors = [COLOR_MAP.get(n, '#888') for n in names]

    fig = go.Figure(go.Bar(
        x=names,
        y=values,
        marker_color=colors,
        text=[f'{v:.4f}' for v in values],
        textposition='outside',
    ))

    if offline_value and offline_value > 0:
        fig.add_hline(
            y=offline_value,
            line_dash='dash',
            line_color='rgba(46,204,113,0.6)',
            annotation_text='Optimal ceiling',
            annotation_position='top right',
        )

    fig.update_layout(
        title='Block Value Captured',
        xaxis_title='Algorithm',
        yaxis_title='Total Block Value',
        showlegend=False,
        height=380,
        **_LAYOUT,
    )
    return fig


def value_history_chart(results, arrival_times=None):
    """Cumulative block value over time / transactions for all algorithms."""
    fig = go.Figure()

    for r in results:
        if not r['history']:
            continue
        n = len(r['history'])
        x = (list(arrival_times[:n]) if arrival_times is not None
             else list(range(n)))
        fig.add_trace(go.Scatter(
            x=x,
            y=r['history'],
            mode='lines',
            name=r['name'],
            line=dict(color=COLOR_MAP.get(r['name'], '#888'), width=2),
        ))

    fig.update_layout(
        title='Block Value Accumulation Over Slot',
        xaxis_title='Time (sec)' if arrival_times is not None else 'Transaction #',
        yaxis_title='Cumulative Block Value',
        height=380,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        **_LAYOUT,
    )
    return fig


def threshold_evolution_chart(thresholds, arrival_times=None):
    """Thompson Sampling acceptance threshold over transaction arrivals."""
    if not thresholds:
        return None
    x = (list(arrival_times[:len(thresholds)]) if arrival_times is not None
         else list(range(len(thresholds))))
    fig = go.Figure(go.Scatter(
        x=x,
        y=thresholds,
        mode='lines',
        name='Threshold',
        line=dict(color=COLOR_MAP['Thompson Sampling'], width=2),
        fill='tozeroy',
        fillcolor='rgba(52,152,219,0.08)',
    ))
    fig.update_layout(
        title='Thompson Sampling: Adaptive Threshold',
        xaxis_title='Time (sec)' if arrival_times is not None else 'Transaction #',
        yaxis_title='Min value/gas to accept',
        height=300,
        showlegend=False,
        **_LAYOUT,
    )
    return fig


def cstar_chart(c_star):
    """c*(b) — optimal shadow price as a function of remaining gas capacity."""
    b_vals = list(range(len(c_star)))
    fig = go.Figure(go.Scatter(
        x=b_vals,
        y=list(c_star),
        mode='lines',
        line=dict(color=COLOR_MAP['Optimal Threshold (c*)'], width=2),
        fill='tozeroy',
        fillcolor='rgba(155,89,182,0.08)',
    ))
    fig.update_layout(
        title='Optimal Threshold c*(b): Shadow Price vs. Remaining Gas',
        xaxis_title='Remaining gas capacity b',
        yaxis_title='c*(b) — min value/gas to accept',
        height=300,
        showlegend=False,
        **_LAYOUT,
    )
    return fig


def race_chart(race_result):
    """Two builders' cumulative block value over time (representative trial)."""
    sa = race_result['sample_a']
    sb = race_result['sample_b']

    fig = go.Figure()

    def _add(res, label, color):
        times = res.get('times', list(range(len(res['history']))))
        history = res['history']
        n = min(len(times), len(history))
        if n == 0:
            return
        fig.add_trace(go.Scatter(
            x=list(times[:n]),
            y=history[:n],
            mode='lines',
            name=label,
            line=dict(color=color, width=2),
        ))

    _add(sa, f"Builder A — {race_result['policy_a']}", COLOR_MAP['Builder A'])
    _add(sb, f"Builder B — {race_result['policy_b']}", COLOR_MAP['Builder B'])

    winner = 'A' if sa['total_value'] >= sb['total_value'] else 'B'
    fig.update_layout(
        title=f'Sample Trial — Builder {winner} wins',
        xaxis_title='Time (sec)',
        yaxis_title='Cumulative Block Value',
        height=380,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        **_LAYOUT,
    )
    return fig


def win_rate_chart(race_result):
    """Donut chart of win rates across all trials."""
    fig = go.Figure(go.Pie(
        labels=[
            f"Builder A\n({race_result['policy_a']})",
            f"Builder B\n({race_result['policy_b']})",
        ],
        values=[race_result['a_win_rate'], race_result['b_win_rate']],
        hole=0.45,
        marker_colors=[COLOR_MAP['Builder A'], COLOR_MAP['Builder B']],
        textinfo='label+percent',
    ))
    fig.update_layout(title='Win Rate Across All Trials', height=320, **_LAYOUT)
    return fig


def value_distribution_chart(a_values, b_values, label_a, label_b):
    """Overlapping histogram of final block values for two builders."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=a_values, name=label_a, opacity=0.65,
        marker_color=COLOR_MAP['Builder A'], nbinsx=30,
    ))
    fig.add_trace(go.Histogram(
        x=b_values, name=label_b, opacity=0.65,
        marker_color=COLOR_MAP['Builder B'], nbinsx=30,
    ))
    fig.update_layout(
        barmode='overlay',
        title='Distribution of Final Block Values',
        xaxis_title='Block Value',
        yaxis_title='Count',
        height=320,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        **_LAYOUT,
    )
    return fig


def flashbots_distribution_chart(raw_values, fitted_lam):
    """Real Flashbots bid values with fitted exponential overlay."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=raw_values, name='Relay bids',
        marker_color='#FFA15A', opacity=0.7,
        nbinsx=40, histnorm='probability density',
    ))
    x_range = np.linspace(0, np.percentile(raw_values, 98), 200)
    y_exp = fitted_lam * np.exp(-fitted_lam * x_range)
    fig.add_trace(go.Scatter(
        x=x_range, y=y_exp,
        mode='lines', name='Fitted Exp(λ)',
        line=dict(color='#FF6692', width=2, dash='dash'),
    ))
    fig.update_layout(
        title='Real Bid Values vs. Fitted Model',
        xaxis_title='Block Value (Gwei)',
        yaxis_title='Density',
        height=300,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        **_LAYOUT,
    )
    return fig


def competitive_ratio_table(results, offline_value):
    """DataFrame of per-algorithm metrics for st.dataframe."""
    rows = []
    for r in results:
        ratio = (r['total_value'] / offline_value
                 if offline_value and offline_value > 0 else float('nan'))
        rows.append({
            'Algorithm': r['name'],
            'Block Value': round(r['total_value'], 4),
            'Competitive Ratio': round(ratio, 4),
            'vs Optimal': f"{ratio * 100:.1f}%",
        })
    return pd.DataFrame(rows)
