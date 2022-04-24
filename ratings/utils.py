"""
Support module for Elo system.
"""

def k_decay(K: float, p1: int, p2: int, alpha: float = 1.5, beta: float = 1.8):
    """
    Most recent iteration of the decay function for head-to-head result weights
    in cycling results. K is the weight given to the race in question, and this
    function updates K for individual head-to-heads based on their overall
    "importance". Head-to-heads are more "important" if they (1) involve the highest
    placed rider(s) in the race or (2) involve riders who placed close to one another.
    K is first scaled based off the absolute place of the higher placed rider (p1), and then
    based on the relative placing (difference between p1 and p2).Adjust alpha and beta 
    to signify different importances of absolute placing (alpha) and relative 
    placing (beta). For both alpha and beta, higher values signify an increase in 
    the rate of decay as absolute/relative placing get larger.
    """

    # adjust K based on absolute placing
    out = K * (p1 / (p1 ** alpha))

    # adjust K based on relative placing
    out = out * ((p2 - p1) / ((p2 - p1) ** beta))

    return out
