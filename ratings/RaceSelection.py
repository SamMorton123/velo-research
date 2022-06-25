"""
Module handling special operations relating to race selection.

April 30, 2022
"""

PERMITTED_TT_LENGTH_KWDS = ['prologue', 'mid-length', 'long', "none", None]
PERMITTED_ITT_VERT_KWDS = ['flat', 'hilly', "none", None]

E = 0.3
FR_MAX = 1
PROLOGUE_DEFAULT_RANGE = (7, 11)
MID_LENGTH_DEFAULT_RANGE = (20, 35)
LONG_TT_DEFAULT_RANGE = (45, 55)
FLAT_TT_DEFAULT_RANGE = (0, 15)
HILLY_TT_DEFAULT_RANGE = (20, 80)


def weight_itt_by_type(race_data, tt_cats, initial_k):
    """
    Enable rankings for different types of TTs (e.g. prologues, long TTs, hilly TTs, etc.)
    by adjusting the initial weight of a TT based its similarity to the given TT type.

    The tt_cats arg is a tuple containing up to two categories. The first specifies the desired
    length of the TT to adjust the weight based on, and the second gives the desired vert of the
    TT to adjust the weight based on. If either is None then the weight will not be adjusted based
    on that metric. Units for first element of tt_cats should be kms, units of second element of
    tt_cats should be meters.
    """

    # basic arg checking
    if len(tt_cats) != 2:
        raise ValueError('Length of tt_cats arg must be 2.')
    if tt_cats[0] not in PERMITTED_TT_LENGTH_KWDS:
        raise ValueError(f'First arg of tt_cats must be in {PERMITTED_TT_LENGTH_KWDS}.')
    if tt_cats[1] not in PERMITTED_ITT_VERT_KWDS:
        raise ValueError(f'Second arg of tt_cats must be in {PERMITTED_ITT_VERT_KWDS}.')

    # make a copy of the given k for adjustment
    adjusted_k = float(initial_k)

    # get relevant data from the race data
    tt_length = race_data['length'].iloc[0]
    tt_vert = race_data['vertical_meters'].iloc[0] / tt_length

    # adjust weight based on given length (if given)
    if tt_cats[0] == 'prologue':
        adjusted_k *= _weight_adjustment_helper(PROLOGUE_DEFAULT_RANGE, tt_length)

    if tt_cats[0] == 'mid-length':
        adjusted_k *= _weight_adjustment_helper(MID_LENGTH_DEFAULT_RANGE, tt_length)
    
    if tt_cats[0] == 'long':
        adjusted_k *= _weight_adjustment_helper(LONG_TT_DEFAULT_RANGE, tt_length)
    
    # adjust weight based on given vert (if given)
    if tt_cats[1] == 'flat':
        adjusted_k *= _weight_adjustment_helper(FLAT_TT_DEFAULT_RANGE, tt_vert)
    
    if tt_cats[1] == 'hilly':
        adjusted_k *= _weight_adjustment_helper(HILLY_TT_DEFAULT_RANGE, tt_vert)
    
    print(f'\nGiven weight: {initial_k}; Adjusted: {adjusted_k}')
    print(f'Length: {tt_length}; Vert: {tt_vert / tt_length}\n')
    return adjusted_k

def _weight_adjustment_helper(rnge, val):

    if val < rnge[0]:
        return (1 / max(FR_MAX, (rnge[0] - val) ** E))
    
    if val > rnge[1]:
        return (1 / max(FR_MAX, (val - rnge[1]) ** E))
    
    return 1

