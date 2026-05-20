import logging
from typing import Callable
import numpy as np
from scipy import ndimage as nd

from ._common_ops import (
    lesion_vol,
    stable_lesion,
    one_dil,
    two_dil,
    three_dil,
    one_ero,
    three_ero,
    two_ero,
    as_new_lesion
)

DEFAULT_STRUCT = nd.generate_binary_structure(3, 3)

logger = logging.getLogger(__name__)


# PLAUSIBLE_PARAMS = {
#     # volumes
#     'big_les': 2500,
#     'smol_les': 250,

#     # unif dist upper bounds
#     'ero': 0.75,
#     'stable': 0.3,
#     '1_ero': 0.65,
#     '2_ero': 0.73,
#     '3_ero': 0.75,
#     'new_les': 0.99,
#     '1_dil': 1.0
# }

# using lambdas here and in CRAZY_PARAMS makes the functions unusable
# with multiprocessing... (cannot be pickled)
PLAUSIBLE_PARAMS = [
    [lambda u, v: u < 0.3 or v > 2500, stable_lesion],
    [lambda u, v: u < 0.75,
        [[lambda u, v: u < 0.65, one_ero],
         [lambda u, v: u < 0.73 or v < 250, two_ero],
         [lambda u, v: u < 1., three_ero]]
    ],
    [[lambda u, v: u < 0.99, as_new_lesion],
     [lambda u, v: u < 1., one_dil]]
]

def _is_list_of_lists(l: list):
    return all(isinstance(ll, list) for ll in l)

def exhaust_list(l: list, **params) -> Callable[[np.ndarray], np.ndarray]:
    if _is_list_of_lists(l):
        for ll in l:
            res = exhaust_list(ll, **params)
            if res is not None:
                return res

    f, ff = l
    dothis = f(**params)
    if dothis and not isinstance(ff, list):
        return ff
    elif dothis:
        return exhaust_list(ff, **params)

def _choice_plausible_params(this_lesion: np.ndarray, vol: float, x: float, params: dict, verbose: bool) -> np.ndarray:

    if verbose: print(f'vol: {vol}, unif: {x}')

    if x < params['stable'] or vol > params['big_les']:
        if verbose: print('Stable lesion.')
        return this_lesion * 1

    if x < params['ero']:
        if x < params['1_ero']:
            if verbose: print('1 ero.')
            return nd.binary_erosion(this_lesion, iterations=1)

        if x < params['2_ero'] or vol < params['smol_les']:
            if verbose: print('2 ero.')
            return nd.binary_erosion(this_lesion, iterations=2)

        if x < params['3_ero']:
            if verbose: print('3 ero.')
            return nd.binary_erosion(this_lesion, iterations=3)

    elif x < params['new_les']:
        if verbose: print('As new lesion.')
        return this_lesion * 0

    else:
        if verbose: print('1 dil.')
        return nd.binary_dilation(this_lesion, iterations=1)

    raise RuntimeError('Wtf')

# CRAZY_PARAMS = {
#     "stable": 0.4,

#     # small lesion augmentations
#     'smol': [200, {
#         "stable": 0.9,
#         "new_les": 1.
#     }],
#     "med": [1000, {
#         "1_ero": 0.5,
#         "1_dil": 0.6,
#         "new_les": 0.7,
#         "stable": 1.
#     }],
#     "big": [np.inf, {
#         "1_ero": 0.55,
#         "1_dil": 0.70,
#         "new_les": 0.85,
#         "stable": 1.
#     }]
# }

CRAZY_PARAMS = [

    [lambda u, v: u < 0.4, stable_lesion],

    [lambda u, v: v < 200, [
        [lambda u, v: u < 0.9, stable_lesion],
        [lambda u, v: u < 1., as_new_lesion]
    ]],

    [lambda u, v: v < 1000, [
        [lambda u, v: u < 0.5, one_ero],
        [lambda u, v: u < 0.6, one_dil],
        [lambda u, v: u < 0.7, as_new_lesion],
        [lambda u, v: u < 1., stable_lesion]
    ]],

    [lambda u, v: v < np.inf, [
        [lambda u, v: u < 0.55, one_ero],
        [lambda u, v: u < 0.70, one_dil],
        [lambda u, v: u < 0.85, as_new_lesion],
        [lambda u, v: u < 1., stable_lesion]
    ]]

]

FRENCH_PARAMS = [
    [lambda u, v: u < 1/8, stable_lesion],
    [lambda u, v: u < 2/8, as_new_lesion],
    [lambda u, v: u < 3/8, one_ero],
    [lambda u, v: u < 4/8, two_ero],
    [lambda u, v: u < 5/8, three_ero],
    [lambda u, v: u < 6/8, one_dil],
    [lambda u, v: u < 7/8, two_dil],
    [lambda u, v: u < 1., three_dil]
]

def _choice_crazy_params(this_lesion: np.ndarray, vol: float, x: float, params: dict) -> np.ndarray:

    log_message = f'vol: {vol}, unif: {x}'
    if x < params['stable']:
        f = stable_lesion

    elif vol < params['smol'][0]:
        smol_params = params['smol'][-1]
        if x < smol_params['stable']:
            f = stable_lesion
        else:
            f = as_new_lesion

    elif vol < params['med'][0]:
        med_params = params['med'][-1]
        if x < med_params['1_ero']:
            f = one_ero
        elif x < med_params['1_dil']:
            f = one_dil
        elif x < med_params['new_les']:
            f = as_new_lesion
        else:
            f = stable_lesion

    elif vol < params['big'][0]:
        big_params = params['big'][-1]
        if x < big_params['1_ero']:
            f = one_ero
        elif x < big_params['1_dil']:
            f = one_dil
        elif x < big_params['new_les']:
            f = as_new_lesion
        else:
            f = stable_lesion
    else:
        raise ValueError(f'Wtf. {vol = }, {x = }')

    return f(this_lesion), log_message + f' - {f.__name__}.'

def _fake_lesion_mask(labeled_lesions_array: np.ndarray, num_lesions: int, spacing, params: list):
    out = np.zeros_like(labeled_lesions_array, dtype=bool)
    for l in range(1, num_lesions + 1):
        logger.debug(f'Processing lesion {l}')

        unif_draw = np.random.uniform()
        this_lesion = labeled_lesions_array == l
        lesion_volume = lesion_vol(this_lesion, spacing)

        op = exhaust_list(params, u=unif_draw, v=lesion_volume)

        out += op(this_lesion)
        logger.debug(f'vol: {lesion_volume}, unif: {unif_draw} - {op.__name__}')

    return out

def fake_lesion_mask(lesion_mask: np.ndarray, spacing, params: list, structure: np.ndarray = DEFAULT_STRUCT) -> np.ndarray:
    labeled_mask, num_lesions = nd.label(lesion_mask, structure=structure)
    return _fake_lesion_mask(labeled_mask, num_lesions, spacing, params)

