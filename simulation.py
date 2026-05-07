# Sealing Problem Simulation
# Dylan Tran (dt2758), Audrey Acken, Colin Calvetti

"""
Block builder sealing problem - variable gas cost version

Each Ethereum transaction has:
    - v - value (ETH tip to builder as incentive)
    - w - gas cost (units consumed from block capacity B)

The builder can accept or reject any incoming transaction.

To optimize a block, the builder wants to maximize v/w (value per gas unit).
At each transaction, there are 2 decisions: include it in the block, or exclude.

We define a threshold to determine which transaction is worth taking, based on
how much gas the builder has remaining.

Data Format:

{
  "slot":           "9500000",
  "block_hash":     "0x3f2a...",
  "builder_pubkey": "0xabc1...",
  "value":          "31250000000000000",  
  "timestamp_ms":   "1710234567891" 
}

Several different core algorithms to test optimal block sealing.

"""

# offline knapsack - optimal in hindsight
def offline_knapsack(values, B):
    pass

# greedy
def greedy():
    pass

#