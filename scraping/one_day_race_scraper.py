import json
import scrape_lib_v2 as scrl

with open('../data/races_data.json') as f:
    RACES = json.load(f)
f.close()

def create_link(name, year, base = 'https://www.procyclingstats.com/race/'):
    return f'{base}{name}/{year}'

for year in range(2007, 2021):
    
    print(f'===== {year} =====')
    
    race_info = []
    for race in RACES['one-day-races'][str(year)]:
        race_info.append((race, create_link(race, year), True))
    
    scrl.get_race_data(race_info, fname = '../data/velodata.csv')