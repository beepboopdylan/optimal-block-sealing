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

    for v, g in zip(values, gas_costs.astype(int)):
        if g > remaining:
            continue

        # estimate current threshold from sampling from posterior mean density
        # with equilibrium condition to accept only if density exceeds average
        # current_threshold = mean_density / (1 + mu/lamda_prior)

        # higher mu (slot ending rate faster) or lower lambda (few arrivals) leads to lower threshold
        mean_density = alpha / beta
        current_threshold = mean_density / (1.0 + mu / lamda_prior)
        thresholds.append(current_threshold)
        
        density = v/g
        
        if density >= current_threshold:
            total += v
            remaining -= g

        # Bayesian update
        # update alpha by 1 since we're accumulating observations, and b by +density 
        # since we're accumulating observed density
        # equivalent to Cianflone's alpha += reward, beta += (1 - reward), but this time
        # for continuous reward instead of a binary

        alpha += 1
        beta += density

    return (total, thresholds)


values   = np.array([6., 4., 3., 2.])
gas_costs = np.array([3,  2,  2,  1])
B = 4

print(offline_knapsack(values, gas_costs, B))   # expect 8.0
print(greedy_unsorted(values, gas_costs, B))     # expect 8.0
print(greedy_sorted(values, gas_costs, B))       # expect 8.0 or close
