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
