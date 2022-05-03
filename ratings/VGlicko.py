"""
Implementation of Glicko-2 ratings for cycling results.

May 2, 2022
"""

from datetime import date
import numpy as np
from termcolor import colored

from ratings import Velo
from ratings.entities import GlickoRider, Race

SCALE_CONSTANT = 173.7178
PLACE_DIFF_CONSTANT = 50
MATCHUP_WEIGHT_SCALE = 0.2


class VGlicko(Velo.Velo):

    def __init__(self, initial_rating: int = 1500, initial_rd: int = 350, 
            intial_volatility: float = 0.06, tau: float = 0.2,
            decay_alpha: float = 1.6, decay_beta: float = 1.5,
            season_turnover_default: float = Velo.SEASON_TURNOVER_DEFAULT, 
            elo_q_base: int = Velo.ELO_Q_BASE, elo_q_exponent_denom: int = Velo.ELO_Q_EXPONENT_DENOM):

        super().__init__(
            decay_alpha = decay_alpha,
            decay_beta = decay_beta,
            season_turnover_default = season_turnover_default,
            elo_q_base = elo_q_base,
            elo_q_exponent_denom = elo_q_exponent_denom
        )

        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.initial_volatility = intial_volatility
        self.tau = tau
    
    def add_new_rider(self, name, age):
        """
        Add a new rider object to the Glicko2 system.
        """

        # check if the given name is already in the system
        rider = self.find_rider(name)

        # if the rider isn't already in the system...
        if rider is None:
            rider = GlickoRider(
                name, self.initial_rating, self.initial_rd, 
                self.initial_volatility, age = age
            )
            self.riders[name] = rider
        
        # otherwise update the age of the rider
        else:
            rider.age = age
        
        return rider
    
    def simulate_race(self, race_name, results, race_weight, timegap_multiplier):
        """
        Apply the affect of the given race to the Glicko system.
        """

        # get race date
        race_date = date(
            year = int(results[Velo.YEAR_COL].iloc[0]),
            month = int(results[Velo.MONTH_COL].iloc[0]),
            day = int(results[Velo.DAY_COL].iloc[0])
        )

        # add race object to list of races
        self.races.append(Race(race_name, race_weight, race_date, None))

        # get Glicko2 scale for each rider in the race from their current Glicko rating and RD
        # volatility included in each tuple but isn't changed,
        # race place included but isn't altered
        # so, each tuple is: (Glicko2 rating, Glicko2 RD, volatility, place)
        g2_ratings = {}
        for i in range(len(results.index)):

            # get rider name, place, and age
            rider = results[Velo.RIDER_COL].iloc[i]
            place = results[Velo.PLACES_COL].iloc[i]
            age = results[Velo.AGE_COL].iloc[i]

            # add rider to the system if they're not already in it
            self.add_new_rider(rider, age)

            g2_ratings[rider] = (
                (self.riders[rider].rating - self.initial_rating) / SCALE_CONSTANT,
                self.riders[rider].rd / SCALE_CONSTANT,
                self.riders[rider].volatility,
                place
            )
        
        # get weight for each matchup
        matchup_weights = {}
        for i in range(len(results.index) - 1):
            for j in range(i + 1, len(results.index)):
                
                rider1 = results[Velo.RIDER_COL].iloc[i]
                rider2 = results[Velo.RIDER_COL].iloc[j]

                matchup_weight = self.k_decay(
                    race_weight,
                    i + 1,
                    j + 1,
                    alpha = self.decay_alpha,
                    beta = self.decay_beta
                ) * MATCHUP_WEIGHT_SCALE

                matchup_weights[(rider1, rider2)] = matchup_weight
                matchup_weights[(rider2, rider1)] = matchup_weight

        # Compute the quantity v. This is the estimated variance of the team’s/player’s 
        # rating based only on game outcomes
        g2_v = {
            rider: self.compute_v(g2_ratings, rider)
            for rider in g2_ratings
        }

        # calculate delta for each rider
        deltas = {rider: self.compute_delta(g2_ratings, g2_v, rider) for rider in g2_ratings}

        # calculate updated volatility for each rider
        new_vols = {rider: self.update_volatility(g2_ratings, g2_v, deltas, rider) for rider in g2_ratings}

        # get new rating and new RD for each rider who competed
        new_ratings = {
            rider: self.new_rating_rd(
                g2_ratings, g2_v, new_vols, matchup_weights, rider
            )
            for rider in g2_ratings
        }

        # get new RD for riders that didn't compete
        new_rds_non_competitors = {
            rider: self.compute_non_competitor_rd_update(rider)
            for rider in self.riders
            if rider not in g2_ratings
        }

        # update rider ratings/RD/volatility for all riders
        for rider in new_ratings:
            self.riders[rider].update_rating(
                race_name, race_weight, race_date,
                new_rating = new_ratings[rider][0],
                new_rd = new_ratings[rider][1],
                new_volatility = new_vols[rider]
            )
        for rider in new_rds_non_competitors:
            self.riders[rider].update_rating(
                race_name, race_weight, race_date,
                new_rd = new_rds_non_competitors[rider]
            )
        
    def print_system(self, curr_year, printing_limit = 50):
        
        # sort the list of rider objects by rating
        riders_sorted = sorted(
            list(self.riders.values()),
            key = lambda rider: rider.rating,
            reverse = True
        )

        for pos, rider in enumerate(riders_sorted):
            
            # only continue printing if the rider rating is above the threshold
            # and we haven't hit the page limit
            if pos + 1 > printing_limit: 
                break

            # rider must have competed in the current year or the year before in order
            # to be printed
            if rider.most_recent_active_year < curr_year - 1:
                continue
                
            rating_change = round(rider.rating_history[-1][0] - rider.rating_history[-2][0], 2)
            if rating_change == 0: rating_change = ''
            else:
                color = 'green' if rating_change > 0 else 'red'
                rating_change = colored(rating_change, color)

            print(
                f'{pos + 1}.', f'{rider.name} - {round(rider.rating, 2)} {rating_change}',
                f'[{round(rider.rd, 2)}, {round(rider.volatility, 2)}]',
                f'({round(rider.rating - (2 * rider.rd), 2)} - {round(rider.rating + (2 * rider.rd), 2)}); ',
                f'Num races: {rider.num_races}', 
                f'Active: {rider.most_recent_active_year}, Age: {rider.age}'
            )

    def new_rating_rd(self, g2_ratings, g2_v, new_vols, matchup_weights, rider):

        # get mu, phi, v, and vol variables for given rider for convenience
        mu = g2_ratings[rider][0]
        phi = g2_ratings[rider][1]
        v = g2_v[rider]
        new_vol = new_vols[rider]

        phi_star = np.sqrt((phi ** 2) + (new_vol ** 2))

        # get new phi
        new_phi_denom = np.sqrt((1 / (phi_star ** 2)) + (1 / v))
        new_phi = 1 / new_phi_denom

        # get new mu
        sigma = 0
        for other in g2_ratings:
            if rider != other:

                # skip other if difference in places is >PLACE_DIFF_CONSTANT
                if abs(g2_ratings[rider][3] - g2_ratings[other][3]) > PLACE_DIFF_CONSTANT:
                    continue

                matchup_weight = matchup_weights[(rider, other)]

                s_j = 1 if g2_ratings[rider][3] < g2_ratings[other][3] else 0
                gphi_j = self.compute_g(g2_ratings[other][1])
                E = self.compute_E(mu, g2_ratings[other][0], g2_ratings[other][1])
    
                sigma += (matchup_weight * gphi_j * (s_j - E))

        new_mu = mu + ((new_phi ** 2) * sigma)

        # convert back to original scale
        new_rating = (SCALE_CONSTANT * new_mu) + self.initial_rating
        new_rd = SCALE_CONSTANT * new_phi

        return (new_rating, new_rd)
    
    def compute_non_competitor_rd_update(self, rider):

        phi = self.riders[rider].rd / SCALE_CONSTANT
        vol = self.riders[rider].volatility

        return np.sqrt(
            (phi ** 2) + (vol ** 2)
        ) * SCALE_CONSTANT
    
    def update_volatility(self, g2_ratings, g2_v, deltas, rider, convergence_tolerance = 0.000001):

        # define variables for the rider's phi, v, delta, and volatility for convenience
        phi = g2_ratings[rider][1]
        v = g2_v[rider]
        delta = deltas[rider]
        vol = g2_ratings[rider][2]

        # get value of a
        a = np.log(vol ** 2)

        # init iterative params A and B
        A = float(a)
        
        if (delta ** 2) > ((phi ** 2) + v):
            B = np.log((delta ** 2) - (phi ** 2) - v)
        else:
            B = self.compute_iterative_B(phi, v, delta, a)
        
        f_A = self.volatility_update_f(phi, v, delta, self.tau, a, A)
        f_B = self.volatility_update_f(phi, v, delta, self.tau, a, B)

        while abs(B - A) > convergence_tolerance:

            num_C = (A - B) * f_A
            denom_C = f_B - f_A
            C = A + (num_C / denom_C)
            f_C = self.volatility_update_f(phi, v, delta, self.tau, a, C)

            # A/f_A update
            if f_B * f_C <= 0:
                A = B
                f_A = f_B
            else:
                f_A /= 2
            
            # B/f_B update
            B = C
            f_B = f_C
        
        return np.exp(A / 2)

    def compute_iterative_B(self, phi, v, delta, a):

        k = 1
        while self.volatility_update_f(
            phi, v, delta, self.tau, a, a - (k * self.tau)
        ) < 0:

            k += 1
        
        return a - (k * self.tau)

    @staticmethod
    def volatility_update_f(phi, v, delta, tau, a, x):

        e_x = np.exp(x)
        
        num1 = e_x * ((delta ** 2) - (phi ** 2) - v - e_x)
        denom1 = 2 * (((phi ** 2) + v + e_x) ** 2)
        frac1 = num1 / denom1

        num2 = x - a
        denom2 = tau ** 2
        frac2 = num2 - denom2

        return frac1 - frac2

    @staticmethod
    def compute_delta(g2_ratings, g2_v, rider):

        # init delta
        delta = 0

        # loop through each competitor of the given rider
        for other in g2_ratings:
            if rider != other:

                # skip other if difference in places is >PLACE_DIFF_CONSTANT
                if abs(g2_ratings[rider][3] - g2_ratings[other][3]) > PLACE_DIFF_CONSTANT:
                    continue

                # get the given rider's "score" against their competitor
                # 1 for victory, 0 for loss
                s_j = 1 if g2_ratings[rider][3] < g2_ratings[other][3] else 0

                gphi_j = VGlicko.compute_g(g2_ratings[other][1])
                E = VGlicko.compute_E(g2_ratings[rider][0], g2_ratings[other][0], g2_ratings[other][1])

                delta += (gphi_j * (s_j - E))
        
        return g2_v[rider] * delta
    
    @staticmethod
    def compute_v(g2_ratings, rider):

        # init sum to be inverted
        sigma = 0

        # loop through each opponent of the given rider in the g2_ratings
        for other in g2_ratings:
            if rider != other:

                # skip other if difference in places is >PLACE_DIFF_CONSTANT
                if abs(g2_ratings[rider][3] - g2_ratings[other][3]) > PLACE_DIFF_CONSTANT:
                    continue

                # get g squared
                gphi_j2 = VGlicko.compute_g(g2_ratings[other][1]) ** 2

                # get E
                E = VGlicko.compute_E(g2_ratings[rider][0], g2_ratings[other][0], g2_ratings[other][1])

                sigma += (gphi_j2 * E * (1 - E))

        return 1 / sigma
    
    @staticmethod
    def compute_g(phi):
        denom = np.sqrt(1 + ((3 * np.power(phi, 2)) / np.power(np.pi, 2)))
        return 1 / denom
    
    @staticmethod
    def compute_E(mu, mu_j, phi_j):
        denom = 1 + np.exp(-VGlicko.compute_g(phi_j) * (mu - mu_j))
        return 1 / denom
