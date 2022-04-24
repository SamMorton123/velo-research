'''
The weight of head-to-heads within a race drops as the winning rider's place goes down
the results list, and as the difference in finishing place between the two riders increases.
I added this code because it seemed intuitive to me that the result of the head to head 
between the rider who wins the race and the rider who gets second should carry greater weight
in altering Elo ratings than the result of the 10th place rider matched up against the 50th
placed rider.
'''

import matplotlib.pyplot as plt
import numpy as np

def logistic_decay(race_weight, rider1_place, rider2_place,
                        steepness = 0.3, inflection_pt = 12):

    return max(0, race_weight / (1 + np.exp((-1 * steepness) * (inflection_pt - rider1_place))))

def linear2(race_weight, rider1_place, rider2_place):
    k = race_weight / np.sqrt(rider1_place) 
    n = k / (rider2_place - rider1_place)
    return n

def linear3(race_weight, rider1_place, rider2_place, zero_threshold = 50):

    place_diff = rider2_place - rider1_place
    if place_diff > zero_threshold:
        return 0

    k = race_weight / np.sqrt(rider1_place) 
    n = k / np.power(place_diff, 1)
    return n

def piecewise(race_weight, rider1_place, rider2_place):
    return (race_weight / np.power(rider1_place, 1.3)) * (rider1_place / rider2_place)

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
