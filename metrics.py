import numpy as np


def metric(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.arccos(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
