from pydantic import BaseModel
from typing import List
import numpy as np

class Point(BaseModel):
    x: float
    y: float
    t: float

class SequenceIn(BaseModel):
    points: List[Point]

    def to_numpy(self) -> np.ndarray:
        # возвращает shape (1,5,3) для GRU модели
        arr = np.array([[p.x, p.y, p.t] for p in self.points], dtype=float)
        return arr.reshape(1, 5, 3)

class PointOut(BaseModel):
    x: float
    y: float
    t: float

    @classmethod
    def from_array(cls, arr: np.ndarray):
        # arr может быть shape (3,) или (1,3) - берем последнюю точку из предсказания
        if len(arr.shape) == 2:
            # Если предсказание последовательности, берем последнюю точку
            arr = arr[-1]
        return cls(x=float(arr[0]), y=float(arr[1]), t=float(arr[2]))
