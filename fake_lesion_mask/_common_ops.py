import numpy as np
import scipy.ndimage as nd

def lesion_vol(lesion_mask: np.ndarray, spacing) -> float:
    return np.prod(spacing) * np.count_nonzero(lesion_mask)

def stable_lesion(lesion_mask: np.ndarray) -> np.ndarray:
    return lesion_mask

def as_new_lesion(lesion_mask: np.ndarray) -> np.ndarray:
    return np.zeros_like(lesion_mask)

def one_ero(lesion_mask: np.ndarray) -> np.ndarray:
    return nd.binary_erosion(lesion_mask, iterations=1)

def two_ero(lesion_mask: np.ndarray) -> np.ndarray:
    return nd.binary_erosion(lesion_mask, iterations=2)

def three_ero(lesion_mask: np.ndarray) -> np.ndarray:
    return nd.binary_erosion(lesion_mask, iterations=3)

def one_dil(lesion_mask: np.ndarray) -> np.ndarray:
    return nd.binary_dilation(lesion_mask, iterations=1)

def two_dil(lesion_mask: np.ndarray) -> np.ndarray:
    return nd.binary_dilation(lesion_mask, iterations=2)

def three_dil(lesion_mask: np.ndarray) -> np.ndarray:
    return nd.binary_dilation(lesion_mask, iterations=3)
