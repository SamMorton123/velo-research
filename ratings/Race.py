class Race:
    def __init__(self, name, weight, datestamp, place, classification = None):

        # init instance variables
        self.name = name
        self.weight = weight
        self.datestamp = datestamp
        self.place = place
        self.classification = classification
    
    def __str__(self):
        return f'{self.name} ({self.classification}); Place: {self.place}'
    
    def __repr__(self):
        return f'{self.name} ({self.classification}); Place: {self.place}'

    def __eq__(self, other_race):
        return self.name == other_race.name and self.datestamp == other_race.datestamp
