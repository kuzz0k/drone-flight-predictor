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
        # возвращает shape (1,5,3)
        arr = np.array([[p.x, p.y, p.t] for p in self.points], dtype=float)
        return arr.reshape(1, 5, 3)

class PointOut(BaseModel):
    x: float
    y: float
    t: float

    @classmethod
    def from_array(cls, arr: np.ndarray):
        return cls(x=arr[0], y=arr[1], t=arr[2])
