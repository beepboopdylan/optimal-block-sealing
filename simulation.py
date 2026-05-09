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