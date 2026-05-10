# Sealing Problem Simulation
# Dylan Tran (dt2758), Audrey Acken, Colin Calvetti

"""
Block builder sealing problem - variable gas cost version

Each Ethereum transaction has:
    - v - value (ETH tip to builder as incentive)
    - w - gas cost (units consumed from block capacity B)

The builder can accept or reject any incoming transaction.

To optimize a block, the builder wants to maximize v/w (value per gas unit).
At each transaction, there are 2 decisions: include it in the block, or save space for a better one.

Mathematical calculations will be using numpy fucntions, and inputs will be numpy lists.
"""

import numpy as np
from scipy.optimize import brentq

# standard knapsack - optimal in hindsight
def offline_knapsack(values, gas_costs, B):
    if len(values) == 0:
        return 0.0

    dp = np.zeros(B + 1)

    for v, g in zip(values, gas_costs.astype(int)):
        if g > B:
            continue
        for b in range(B, g-1, -1):
            dp[b] = max(dp[b], dp[b-g] + v)

    return float(dp[-1])

# greedy upon arrival, without any sorting - naive baseline
def greedy_unsorted(values, gas_costs, B):
    remaining = B
    total = 0.0

    for v, g in zip(values, gas_costs.astype(int)):
        if g <= remaining:
            total += v # include
            remaining -= g
    return total

# greedy sorted by v/w in descending order.
def greedy_sorted(values, gas_costs, B):
    if len(values) == 0:
        return 0.0

    densities = values / gas_costs
    order = np.argsort(densities)[::-1]

    remaining = B
    total = 0.0

    for d in order:
        g = int(gas_costs[d])
        if g <= remaining:
            total += values[d]
            remaining -= g
    return total

# Theoretically optimal threshold benchmark with Poisson

def compute_threshold(B, lam, mu, n_samples=40_000):
    """

    Computing theoretically optimal threshold table c*(b) by solving
    fixed-point equation derived from the Bellman equation for
    Poisson deadline stopping model, mentioned in OH by Prof. Roughgarden:

    mu * W(b) = lamda * E[ max(v - g*c*(b), 0) * 1(g <= b) ]
    c*(b) = W(b) - W(b-1)

    Poisson deadline:
        Since exponential distribution is memoryless, the remaining slot
        lifetime looks identical at every moment. Time is no longer an
        important factor. c*(b) now only depends on gas remaining in b,
        not on time.

    We then need to know upfront lambda and value of distribution.

    """
    v_samples = np.random.exponential(1.0, n_samples)
    w_samples = np.random.randint(1, 6, n_samples).astype(float)

    W = np.zeros(B + 1)

    for b in range(1, B + 1):
        W_prev = W[b - 1]

        def equation(delta):
            # delta = c*(b) = shadow price per gas unit
            # accept (v,w) iff v/w >= delta, i.e. v - w*delta >= 0
            # only if transaction fits: w <= b
            feasible = w_samples <= b
            gain = np.mean(np.maximum(v_samples - w_samples * delta, 0) * feasible)
            return mu * (W_prev + delta) - lam * gain

        lo = 0.0
        hi = float(np.max(v_samples / w_samples))

        # Bisection works because:
        #   LHS mu*(W_prev + delta) is strictly increasing in delta
        #   RHS lam*E[...] is strictly decreasing in delta
        #   Unique crossing guaranteed by intermediate value theorem
        W[b] = W_prev + (0.0 if equation(lo) >= 0
                         else brentq(equation, lo, hi, xtol=1e-8))

    # c_star[b] = W[b] - W[b-1] = shadow price at each capacity level
    # Non-increasing in b: be more selective with more gas remaining
    c_star = np.diff(W, prepend=0.0)
    return c_star

def apply_threshold(values, gas_costs, c_star, B):
    """
    Apply precomputed threshold policy to a transaction stream.
    Accept (v, w) if it fits AND v/w >= c_star[remaining_gas].
    """
    remaining = B
    total = 0.0
    for v, g in zip(values, gas_costs.astype(int)):
        if g <= remaining and (v / g) >= c_star[remaining]:
            total += v
            remaining -= g
    return total


"""

Thompson Sampling

This method is applicable to the optimal sealing problem because of several reasons:
    1. Builders face an online decision problem -- transactions arrive in stream, with irreversible choices
    2. They don't know the value distribution beforehand. It's ever changing and random

Thompson Sampling learns the distribution in real time and adapts the threshold as it observes transactions.

We will be building off of the standard Beta-Bernoulli formulation in https://github.com/andrecianflone/thompson:
    1. The rewards are continuous as the mempool is dynamic, not binary.
        - Reward is the density v/w > 0, modeled with Gamma(alpha, beta) instead of Beta
        - Gamma instead of Beta for continuity, and is conjugate prior for exponential data, making
        Bayesian updates O(1): alpha += 1, beta += observed_density
    2. No arm selection
        - At each arriving transaction, decide accept or reject
        - Sample from current posterior to estimate threshold, and accept if density >= threshold
    3. Dynamic threshold
        - The earlier slots have fewer observations, so higher uncertainty and more unstable threshold
        - The later slots have more observations, so lower uncertainty and more stable threshold

"""

def thompson_sampling(values, gas_costs, B, mu, lamda_prior=1.0, prior_alpha=1.0, prior_beta=1.0):
    """
    values, gas_costs: transaction stream o arrival
    B: block capacity
    mu: how fast the slot ends (Poisson deadline)
    lamda_prior: prior belief on arrival rate lambda
    prior_alpha: Gamma prior shape (default = 1.0)
    prior_beta: Gamma prior rate (default = 1.0)

    Outputs (total, thresholds (for plotting))

    """
    if len(values) == 0:
        return 0.0, []

    thresholds = []
    remaining = B

    # uninformative priors
    alpha = prior_alpha
    beta = prior_beta

    total = 0.0

    for i, (v, g) in enumerate(zip(values, gas_costs.astype(int))):
        # online lambda estimate: after i+1 transactions, lambda >= (i+1)*mu
        # (each transaction represents ~1/mu elapsed time by the memoryless property,
        # so n transactions implies arrival rate >= n*mu)
        lamda_est = max(lamda_prior, (i + 1) * mu)

        if g > remaining:
            continue

        # higher mu (slot ending rate faster) or lower lambda (few arrivals) leads to lower threshold
        mean_density = beta / alpha
        current_threshold = mean_density / (1.0 + mu / lamda_est)
        thresholds.append(current_threshold)

        density = v / g

        if density >= current_threshold:
            total += v
            remaining -= g

        alpha += 1
        beta += density

    return (total, thresholds)


# ── Stream generation ─────────────────────────────────────────────────────────

def generate_stream(lam, T=12.0, value_dist='exponential', value_mean=10.0,
                    gas_mean=20, seed=None):
    """
    Generate a Poisson transaction stream over slot duration T seconds.
    Returns (values, gas_costs, arrival_times).
    """
    rng = np.random.default_rng(seed)
    n = rng.poisson(lam * T)
    arrival_times = np.sort(rng.uniform(0, T, n))

    if value_dist == 'exponential':
        values = rng.exponential(value_mean, n)
    elif value_dist == 'pareto':
        values = (rng.pareto(2.0, n) + 1) * value_mean * 0.5
    elif value_dist == 'lognormal':
        sigma = 1.0
        mu_ln = np.log(max(value_mean, 1e-9)) - 0.5 * sigma ** 2
        values = rng.lognormal(mu_ln, sigma, n)
    else:
        values = rng.exponential(value_mean, n)

    gas_costs = rng.integers(max(1, gas_mean // 4), gas_mean + 1, n).astype(float)
    return values, gas_costs, arrival_times


# ── Variants with per-transaction history (for time-series charts) ────────────

def greedy_online(values, gas_costs, B):
    """Accept first-come-first-served if it fits. Returns (total, history)."""
    remaining = int(B)
    total = 0.0
    history = []
    for v, g in zip(values, gas_costs.astype(int)):
        if g <= remaining:
            total += v
            remaining -= g
        history.append(total)
    return total, history


def greedy_density(values, gas_costs, B):
    """Sort by value/gas density then accept greedily. Returns (total, history)."""
    if len(values) == 0:
        return 0.0, []
    order = np.argsort(values / gas_costs)[::-1]
    remaining = int(B)
    accepted = np.zeros(len(values), dtype=bool)
    total = 0.0
    for d in order:
        g = int(gas_costs[d])
        if g <= remaining:
            total += values[d]
            remaining -= g
            accepted[d] = True
    running = 0.0
    history = []
    for i in range(len(values)):
        if accepted[i]:
            running += values[i]
        history.append(running)
    return total, history


def _ts_with_history(values, gas_costs, B, mu, lamda_prior=1.0,
                     prior_alpha=1.0, prior_beta=1.0):
    """Thompson Sampling with per-transaction history for charting."""
    if len(values) == 0:
        return 0.0, [], []
    remaining = int(B)
    alpha, beta = prior_alpha, prior_beta
    total = 0.0
    history = []
    thresholds = []

    for i, (v, g) in enumerate(zip(values, gas_costs.astype(int))):
        lamda_est = max(lamda_prior, (i + 1) * mu)

        if g > remaining:
            history.append(total)
            continue

        mean_density = beta / alpha
        threshold = mean_density / (1.0 + mu / lamda_est)
        thresholds.append(threshold)

        if v / g >= threshold:
            total += v
            remaining -= g

        alpha += 1
        beta += v / g
        history.append(total)

    return total, history, thresholds


def _apply_threshold_with_history(values, gas_costs, c_star, B):
    """apply_threshold with per-transaction history for charting."""
    remaining = int(B)
    total = 0.0
    history = []
    for v, g in zip(values, gas_costs.astype(int)):
        if g <= remaining and (v / g) >= c_star[remaining]:
            total += v
            remaining -= g
        history.append(total)
    return total, history


# ── Unified runner ────────────────────────────────────────────────────────────

ALGORITHMS = [
    'Greedy (online)',
    'Thompson Sampling',
    'Optimal Threshold (c*)',
    'Optimal (offline)',
    'Greedy (density-sorted)',
]


def run_algorithm(name, values, gas_costs, B, mu=0.083, lam_prior=10.0,
                  c_star=None):
    """
    Run a named algorithm and return a uniform result dict.

    Pass c_star to skip recomputing it for 'Optimal Threshold (c*)' —
    useful when running many trials in a race.
    """
    thresholds = []
    c_star_out = None

    if name == 'Greedy (online)':
        total, history = greedy_online(values, gas_costs, B)

    elif name == 'Thompson Sampling':
        total, history, thresholds = _ts_with_history(
            values, gas_costs, B, mu, lamda_prior=lam_prior)

    elif name == 'Optimal Threshold (c*)':
        if c_star is None:
            c_star = compute_threshold(int(B), lam_prior, mu)
        total, history = _apply_threshold_with_history(values, gas_costs, c_star, B)
        c_star_out = c_star

    elif name == 'Optimal (offline)':
        total = offline_knapsack(values, gas_costs, int(B))
        history = [total] * len(values)

    elif name == 'Greedy (density-sorted)':
        total, history = greedy_density(values, gas_costs, B)

    else:
        raise ValueError(f"Unknown algorithm: {name}")

    return {
        'name': name,
        'total_value': float(total),
        'history': history,
        'thresholds': thresholds,
        'c_star': c_star_out,
    }


# ── Two-builder competition ───────────────────────────────────────────────────

def two_builder_race(policy_a, policy_b, lam, T, B, mu,
                     value_dist='exponential', value_mean=10.0,
                     gas_mean=20, n_trials=200, seed=0):
    """
    Simulate n_trials of two builders competing with independent Poisson streams.
    Winner-take-all: whoever seals the higher block value wins the slot.

    Returns aggregate stats plus a representative sample trial for visualization.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2 ** 31, (n_trials, 2))

    # Precompute c* once if either policy needs it
    precomputed_cstar = None
    if 'Optimal Threshold (c*)' in (policy_a, policy_b):
        precomputed_cstar = compute_threshold(int(B), lam, mu)

    a_values, b_values = [], []
    a_wins = 0

    for i in range(n_trials):
        vals_a, gas_a, times_a = generate_stream(
            lam, T, value_dist, value_mean, gas_mean, seed=int(seeds[i, 0]))
        vals_b, gas_b, times_b = generate_stream(
            lam, T, value_dist, value_mean, gas_mean, seed=int(seeds[i, 1]))

        res_a = run_algorithm(policy_a, vals_a, gas_a, B, mu=mu,
                              lam_prior=lam, c_star=precomputed_cstar)
        res_b = run_algorithm(policy_b, vals_b, gas_b, B, mu=mu,
                              lam_prior=lam, c_star=precomputed_cstar)

        a_values.append(res_a['total_value'])
        b_values.append(res_b['total_value'])
        if res_a['total_value'] > res_b['total_value']:
            a_wins += 1

    # Pick representative trial: closest to median of builder A
    a_arr = np.array(a_values)
    rep = int(np.argmin(np.abs(a_arr - np.median(a_arr))))

    vals_a, gas_a, times_a = generate_stream(
        lam, T, value_dist, value_mean, gas_mean, seed=int(seeds[rep, 0]))
    vals_b, gas_b, times_b = generate_stream(
        lam, T, value_dist, value_mean, gas_mean, seed=int(seeds[rep, 1]))
    sample_a = {
        **run_algorithm(policy_a, vals_a, gas_a, B, mu=mu,
                        lam_prior=lam, c_star=precomputed_cstar),
        'times': times_a,
    }
    sample_b = {
        **run_algorithm(policy_b, vals_b, gas_b, B, mu=mu,
                        lam_prior=lam, c_star=precomputed_cstar),
        'times': times_b,
    }

    return {
        'policy_a': policy_a,
        'policy_b': policy_b,
        'a_win_rate': a_wins / n_trials,
        'b_win_rate': (n_trials - a_wins) / n_trials,
        'a_mean_value': float(np.mean(a_values)),
        'b_mean_value': float(np.mean(b_values)),
        'a_std': float(np.std(a_values)),
        'b_std': float(np.std(b_values)),
        'all_a_values': a_values,
        'all_b_values': b_values,
        'sample_a': sample_a,
        'sample_b': sample_b,
    }


if __name__ == '__main__':
    values = np.array([6., 4., 3., 2.])
    gas_costs = np.array([3., 2., 2., 1.])
    B = 4

    print(offline_knapsack(values, gas_costs, B))       # 8.0
    print(greedy_unsorted(values, gas_costs, B))         # 8.0
    print(greedy_sorted(values, gas_costs, B))           # 8.0
