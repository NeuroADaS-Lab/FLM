from __future__ import annotations
import logging
import numpy as np
from scipy import ndimage as nd

from ._common_ops import stable_lesion, as_new_lesion

logger = logging.getLogger(__name__)


def random_iters(iters: list[int], probs: list[float]):
    def wrap(operation):
        def wrapped_f(x: np.ndarray, **kwargs) -> np.ndarray:
            iterations = np.random.choice(iters, p=probs)
            logger.debug('Performing operation %s with %i iterations', operation.__name__, iterations)
            return operation(x, iterations=iterations, **kwargs)
        return wrapped_f
    return wrap


@random_iters(iters=[1, 2, 3], probs=[8/12, 3/12, 1/12])
def erosion(x: np.ndarray, **kwargs):
    return nd.binary_erosion(x, **kwargs)

@random_iters(iters=[1, 2, 3], probs=[8/12, 3/12, 1/12])
def dilation(x: np.ndarray, **kwargs):
    return nd.binary_dilation(x, **kwargs)


OPTIONS = [
    stable_lesion,
    as_new_lesion,
    erosion,
    dilation
]

def _fake_lesion_mask(labeled_lesions_array: np.ndarray, num_lesions: int, params: list):
    out = np.zeros_like(labeled_lesions_array, dtype=bool)
    for l in range(1, num_lesions + 1):
        # print(f'Processing lesion {l}')
        # unif_draw = np.random.uniform()
        # lesion_volume = lesion_vol(this_lesion, spacing)
        out += np.random.choice(params)(labeled_lesions_array == l)
        # print(f'vol: {lesion_volume}, unif: {unif_draw} - {op.__name__}')
    return out

def fake_lesion_mask(lesion_mask: np.ndarray, params: list, structure) -> np.ndarray:
    labeled_mask, num_lesions = nd.label(lesion_mask, structure=structure)
    return _fake_lesion_mask(labeled_mask, num_lesions, params)
