from __future__ import annotations

import math
from typing import Iterable


def dot_product(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(a, b))



def l2_norm(vector: Iterable[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))



def cosine_similarity(a: list[float], b: list[float]) -> float:
    denom = l2_norm(a) * l2_norm(b)
    if denom == 0:
        return 0.0
    return dot_product(a, b) / denom
