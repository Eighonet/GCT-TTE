import numpy as np


def inference(edges, model: str = 'random') -> float:
    if model == 'random':
        return len(edges) * (25 + np.random.randint(75)) / 100
