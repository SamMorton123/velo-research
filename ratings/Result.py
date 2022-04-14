'''
Simple class for storing data regarding a particular rider's result in
a given race.
'''

class Result:
    def __init__(self, name, place, timegap = None):
        self.name = name
        self.place = place
        self.timegap = timegap
