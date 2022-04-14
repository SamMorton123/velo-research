'''
Methods to help elegantly alter the riders that are output when the Elo system is printed.
This code allows the system to print things like the top U23 riders, top GC riders within
TT rankings, etc.
'''

# constants
ESTABLISHED_GC = [
    'POGAČAR Tadej', 'ROGLIČ Primož', 'BERNAL Egan',
    'CARAPAZ Richard', 'MAS Enric', 'VINGEGAARD Jonas',
    'YATES Adam', 'KELDERMAN Wilco', 'PORTE Richie', 'THOMAS Geraint', 'YATES Simon', 'LANDA Mikel',
    'CARUSO Damiano', 'MARTIN Guillaume', "O'CONNOR Ben", 'URÁN Rigoberto', 'Alejandro Valverde',
    'LÓPEZ Miguel Ángel', 'BILBAO Pello', 'HAIG Jack', 'ALMEIDA João', 'QUINTANA Nairo', 'VLASOV Aleksandr',
    'LUTSENKO Alexey', 'KRUIJSWIJK Steven', 'GAUDU David', 'DUMOULIN Tom', 'BARDET Romain', 'FUGLSANG Jakob',
    'CARTHY Hugh', 'GEOGHEGAN HART Tao', 'NIBALI Vincenzo', 'MARTIN Dan', 'IZAGIRRE Ion', 'MARTÍNEZ Daniel Felipe',
    'CHAVES Esteban', 'MAJKA Rafał', 'SCHACHMANN Maximilian', 'KUSS Sepp', 'POELS Wout', 'HINDLEY Jai',
    'BUCHMANN Emanuel', 'DE LA CRUZ David', 'MÄDER Gino', 'MOLLEMA Bauke', 'CATTANEO Mattia', 'BENNETT George',
    'BARGUIL Warren', 'PINOT Thibaut'
]

def select_all(rider):
    '''
    No matter the rider, return True so that the entire
    system is printed.
    '''

    return True

def select_u26(rider):
    return rider.age < 26

def select_established_gc(rider):
    '''
    Given the list of 'established' GC guys, return True if the given
    rider is contained within that list.
    '''
    
    return rider.name in ESTABLISHED_GC