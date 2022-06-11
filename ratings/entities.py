'''
Generic rider class intended to be useful for a cycling Elo implementation
in addition to more sophisticated rating systems in the future.
'''

from datetime import date

# constants
DEFAULT_INITIAL_RATING = 1500
NEW_SEASON_RATING_KEYWORD = 'newseason'

class Rider:
    def __init__(self, name, initial_rating = DEFAULT_INITIAL_RATING, team = None, age = None):
        
        # init instance variables
        self.name = name
        self.rating = initial_rating
        self.team = team
        self.age = age
        self.delta = 0
        self.rating_history = [(self.rating, NEW_SEASON_RATING_KEYWORD)]
        self.race_history = []
        self.most_recent_active_year = 1900
    
    def add_new_race(self, race_name, race_weight, race_date, place):
        new_race = Race(
            race_name,
            race_weight,
            race_date,
            place,
        )
        if new_race not in self.race_history:
            self.race_history.append(new_race)
    
    def update_delta(self, d):
        self.delta += d
    
    def resolve_delta(self, race_name, race_weight, datestamp):
        """
        The Elo system calculates changes in scores via head-to-head results. However,
        the head-to-head matchups are treated as if they occur simultaneously. The delta
        tracks the change in a rider's rating across all head to heads which happen
        in a race, and then this method adds the change to the rider's elo and resets
        the delta.
        """

        # update the most recent year of competition for the rider, if the delta != 0
        if self.delta != 0:
            self.most_recent_active_year = datestamp.year

        # add delta to rating
        self.rating += self.delta

        # reset delta
        self.delta = 0

        # add new rating to rating history
        self.rating_history.append((self.rating, datestamp))
    
    def new_season(self, year, weight):
        """
        There can be unpredictible changes to rider abilities in the off-season, and this
        method is intended to model this at the rider level. Basically, at the end of a season,
        each rider's rating is adjusted to be closer to the default initial rating.
        """

        # get the new rating
        new_rating = (DEFAULT_INITIAL_RATING * weight) + (self.rating * (1 - weight))
        
        # add the new rating to the rider's rating history as a new Race
        self.race_history.append(Race('New Season', None, None, None))
        
        # update the rating
        self.rating = new_rating
        self.rating_history.append((self.rating, date(year = year, month = 1, day = 1)))
    
    def __eq__(self, other_rider):
        return self.name == other_rider.name
    
    def __str__(self):
        return f'{self.name}: {self.rating}'
    
    def __repr__(self):
        return str(self)

class Race:
    def __init__(self, name, weight, datestamp, place):

        # init instance variables
        self.name = name
        self.weight = weight
        self.datestamp = datestamp
        self.place = place
    
    def __str__(self):
        return f'{self.name}: Place: {self.place}'
    
    def __repr__(self):
        return f'{self.name}: Place: {self.place}'

    def __eq__(self, other_race):
        return self.name == other_race.name and self.datestamp == other_race.datestamp
