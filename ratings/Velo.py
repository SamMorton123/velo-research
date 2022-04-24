"""
New gen of the cycling elo class. Changing the name to Velo (V-Elo), and consolidating
functionality to make it a much more coherent module.
"""

from datetime import date, timedelta
import numpy as np
import pandas as pd
from termcolor import colored

# local
from ratings import entities
from ratings.entities import Rider, Race
from ratings.utils import k_decay

# ====== Column name constants ======
PLACES_COL = 'place'
RIDER_COL = 'rider'
AGE_COL = 'age'
TIME_COL = 'time'
YEAR_COL = 'year'
MONTH_COL = 'month'
DAY_COL = 'day'

# ===== Elo system constants ======
ELO_Q_BASE = 10
ELO_Q_EXPONENT_DENOM = 400

# ===== Constant defaults =====
ALPHA_DEFAULT = 1.5
BETA_DEFAULT = 1.8
SEASON_TURNOVER_DEFAULT = 0.4

PRINT_RIDER_NUM_TOP_PLACINGS = 1
TOP_PLACING_DEFINITION = 10

class Velo:
    
    def __init__(self, decay_alpha: float = 1.5, decay_beta: float = 1.8,
            season_turnover_default: float = SEASON_TURNOVER_DEFAULT, 
            elo_q_base: int = ELO_Q_BASE, elo_q_exponent_denom: int = ELO_Q_EXPONENT_DENOM):

        # collect args as instance variables
        self.decay_alpha = decay_alpha
        self.decay_beta = decay_beta
        self.season_turnover_default = season_turnover_default
        self.elo_q_base = elo_q_base
        self.elo_q_exponent_denom = elo_q_exponent_denom
         
        # init dict and array to track rider and race objects respectively
        self.riders = {}  # format: {rider name: rider object}
        self.races = []

        # init dictionary to track rating data over time
        self.rating_data = {'year': [], 'month': [], 'day': []}
    
    def find_rider(self, name):
        """
        Return an instance of the Rider class if the given name is found. Otherwise,
        return None.
        """

        if name in self.riders:
            return self.riders[name]
        
        return None
    
    def add_new_rider(self, name, age):
        """
        Add a new rider object to the Elo system.
        """

        # check if the given name is already in the system
        rider = self.find_rider(name)

        # if the rider isn't already in the system...
        if rider is None:
            rider = Rider(name, age = age)
            self.riders[name] = rider
        
        # otherwise update the age of the rider
        else:
            rider.age = age
        
        return rider
    
    def simulate_race(self, race_name, results, race_weight, timegap_multiplier, k_func = k_decay):
        '''
        Simulate a race as a series of head to head matchups between all participants in the race. This type
        of treatment can be used on different cycling disciplines, such as GC results and TT results.
        '''

        # get race date
        year = int(results[YEAR_COL].iloc[0])
        month = int(results[MONTH_COL].iloc[0])
        day = int(results[DAY_COL].iloc[0])
        race_date = date(year = year, month = month, day = day)

        # add race to list of races
        race_info = Race(race_name, race_weight, race_date, None)
        self.races.append(race_info)

        # init list of rider order
        rider_order = []

        # run each head-to-head
        for i in range(len(results.index) - 1):
            for j in range(i + 1, len(results.index)):

                # get rider i and j places
                rider1_place = results[PLACES_COL].iloc[i]
                rider2_place = results[PLACES_COL].iloc[j]
                
                # if rider i's place isn't less than j's, continue
                if rider1_place >= rider2_place:
                    continue

                # get rider i and j names
                rider1_name = results[RIDER_COL].iloc[i]
                try: rider1_age = int(results[AGE_COL].iloc[i])
                except: rider1_age = 0
                rider2_name = results[RIDER_COL].iloc[j]
                try: rider2_age = int(results[AGE_COL].iloc[j])
                except: rider2_age = 0

                # get rider objects
                rider1 = self.add_new_rider(rider1_name, rider1_age)
                rider2 = self.add_new_rider(rider2_name, rider2_age)

                # add rider i to order if not already in it
                if len(rider_order) == 0 or rider1 != rider_order[-1]:
                    rider_order.append(rider1)

                # add to rider race histories
                rider1.add_new_race(race_name, race_weight, race_date, rider1_place)
                rider2.add_new_race(race_name, race_weight, race_date, rider2_place)

                # get rider times
                try:
                    if rider1_place == 0:
                        rider1_time = timedelta(seconds = 0)
                    else:
                        rider1_time = timedelta(seconds = int(results[TIME_COL].iloc[i]))
                    rider2_time = timedelta(seconds = int(results[TIME_COL].iloc[j]))
                except:
                    rider1_time = 0
                    rider2_time = 0

                # get the matchup weight
                matchup_weight = k_func(race_weight, rider1_place, rider2_place)

                # run the head to head
                self.h2h(
                    rider1, rider1_time, rider2, rider2_time, matchup_weight, timegap_multiplier
                )

        return rider_order

    def h2h(self, rider1, rider1_time, rider2, rider2_time, matchup_weight, timegap_multiplier):
        '''
        Simulate the head to head matchup between two given riders, and adjust their elo deltas
        accordingly.
        '''

        # get respective win/loss probabilities
        rider1_p, rider2_p = _get_win_loss_probabilities(rider1, rider2)

        # set rider "scores" (assumes rider1 beat rider2 if times are equal)
        rider1_score = 1 if rider1_time <= rider2_time else 0
        rider2_score = 1 if rider2_time < rider1_time else 0

        # get margin of victory multiplier
        margvict_mult = _get_margvict_mult(rider1, rider1_time, rider2, rider2_time, timegap_multiplier)

        # get elo deltas for each rider
        rider1_delta = matchup_weight * margvict_mult * (rider1_score - rider1_p)
        rider2_delta = matchup_weight * margvict_mult * (rider2_score - rider2_p)

        # update rider elo deltas
        rider1.update_delta(rider1_delta)
        rider2.update_delta(rider2_delta)

        # update wins/losses
        if rider1_score == 1: rider1.increment_wins()
        else: rider1.increment_losses()
        if rider2_score == 1: rider2.increment_wins()
        else: rider2.increment_losses()

    def apply_all_deltas(self, race_name, race_weight, datestamp):
        for name in self.riders:
            self.riders[name].resolve_delta(race_name, race_weight, datestamp)

    def new_season_regression(self, year, regression_to_mean_weight = 0.5):

        self.races.append(Race('New Season', None, None, None))
        for name in self.riders:
            self.riders[name].new_season(regression_to_mean_weight)
        
        self.save_system(date(year = year, month = 1, day = 1))
    
    def save_system(self, date):
        """
        Save rating data as a timeseries for each rider in the system.
        """

        # loop through every rider in the system and add their current rating to the cumulative data
        for name in self.riders:
            
            if name in self.rating_data:
                self.rating_data[name].append(self.riders[name].rating)
            
            # add a new key and array to the rating_data dict, and make sure that all keys
            # have corresponding arrays of the same length by filling the array with the
            # default initial Elo value and then adding their most recent rating
            else:
                self.rating_data[name] = [
                    entities.DEFAULT_INITIAL_RATING
                ] * len(self.rating_data['year']) + [self.riders[name].rating]
        
        # add a new date to the dict
        self.rating_data['year'].append(date.year)
        self.rating_data['month'].append(date.month)
        self.rating_data['day'].append(date.day)
    
    def save_system_data(self, rating_type, base_fname = 'data/system_data'):
        pd.DataFrame(data = self.rating_data).to_csv(f'{base_fname}_{rating_type}.csv', index = False)

    def print_system(self, curr_year, min_rating = entities.DEFAULT_INITIAL_RATING, max_places = 50):
        '''
        Print the rankings as they currently stand.
        '''
        
        # sort the list of rider objects by rating
        riders_sorted = sorted(
            list(self.riders.values()),
            key = lambda rider: rider.rating,
            reverse = True
        )

        # now print each rider and their place
        place = 1
        for rider in riders_sorted:
            if rider.rating < min_rating or place > max_places: 
                break

            # continue, if the rider hasn't competed in the current year or the year before
            if rider.most_recent_active_year < curr_year - 1:
                continue
            
            # continue, if the rider hasn't placed high enough a sufficient number of times
            sorted_results = sorted(rider.race_history, key = lambda race: (race.place is None, race.place))
            if (len(sorted_results) < PRINT_RIDER_NUM_TOP_PLACINGS or
                sorted_results[PRINT_RIDER_NUM_TOP_PLACINGS - 1].place is None or
                sorted_results[PRINT_RIDER_NUM_TOP_PLACINGS - 1].place > TOP_PLACING_DEFINITION):
                continue

            # get the rider's delta, and add color
            delta = round(rider.rating_history[-1][0] - rider.rating_history[-2][0], 2)
            if delta == 0:
                delta = ''
            else:
                color = 'green' if delta > 0 else 'red'
                delta = colored(delta, color)

            print(
                f'{place}.', f'{rider.name} - {round(rider.rating, 2)};', 
                f'Active: {rider.most_recent_active_year}, Age: {rider.age}', delta,
                f'Top {TOP_PLACING_DEFINITION}s:', len([r for r in sorted_results if r.place is not None and r.place <= TOP_PLACING_DEFINITION])
            )

            place += 1
    
    def print_most_improved(self, riders_printed = 20):
        '''
        Print the most improved riders as a percentage of their rating at the start
        of the season.
        '''

        print('\nMost Improved:\n')
        

        # get riders and their percentage improvements
        riders_p_improvements = [
            (rider, _get_improvement(rider)) for rider in self.riders
        ]

        # sort by improvement
        riders_p_improvements.sort(key = lambda t: t[1], reverse = True)
        for i in range(min(riders_printed, len(riders_p_improvements))):
            print(f'{i + 1}. {riders_p_improvements[i][0].name} - ({riders_p_improvements[i][1]}%)')

    
    @staticmethod
    def top_probabilities(existing_probabilities, rider_order):
        '''
        Still in development. Goal: generating probabilities of a rider doing well in a race
        based on their rating and the ratings of the other competitors in the race.
        '''

        for quality_thresh in existing_probabilities:
            for thresh in existing_probabilities[quality_thresh]:
                for rider in rider_order:
                    if rider.rating >= quality_thresh:
                        if rider_order.index(rider) < thresh:
                            existing_probabilities[quality_thresh][thresh].append(1)
                        else:
                            existing_probabilities[quality_thresh][thresh].append(0)
        return existing_probabilities

def _get_win_loss_probabilities(rider1, rider2):
    '''
    Utility method for helping with calculating rider Elos.
    '''

    # get each competitor's Q value
    rider1_q = np.power(ELO_Q_BASE, rider1.rating / ELO_Q_EXPONENT_DENOM)
    rider2_q = np.power(ELO_Q_BASE, rider2.rating / ELO_Q_EXPONENT_DENOM)

    # get the probabilities and return
    rider1_p = rider1_q / (rider1_q + rider2_q)
    rider2_p = rider2_q / (rider1_q + rider2_q)
    return (rider1_p, rider2_p)

def _get_margvict_mult(rider1, rider1_time, rider2, rider2_time, time_gap_multiplier):
    '''
    ASSUMES rider1 is the winning rider and rider2 is the losing rider. Calculates a multiplier
    to have rating changes take margin of victory into account.
    '''

    # expand time difference with multiplier
    score_diff = (rider1_time - rider2_time) * time_gap_multiplier
    
    # get difference in ratings
    rating_diff = rider1.rating - rider2.rating
    
    # try to calculate the score multiplier
    try:
        return math.log(score_diff + 1) * (2.2 / ((rating_diff * 0.001) + 2.2))
    
    # if there is a problem, just return 1 so that the multiplier simply has no effect on calculations
    except:
        return 1
    
def _get_improvement(rider):

    # get rating from beginning of year
    for i in range(len(rider.rating_history) - 1, -1, -1):
        if rider.rating_history[i][1] == 'newseason':
            initial_rating = rider.rating_history[i][0]
            break

    return round((rider.rating - initial_rating) / initial_rating * 100, 2)