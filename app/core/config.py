from pydantic_settings import BaseSettings
from typing import List, Union
import numpy as np
import ast

class Settings(BaseSettings):
    PROJECT_NAME: str = "drone-flight-predictor"
    VERSION: str = "1.0.0"
    MODEL_PATH: str = "training/checkpoints/best.pt"
    DEVICE: str = "cpu"
    MEAN: Union[str, List[float]] = "[0.0, 0.0, 0.0]"
    STD: Union[str, List[float]] = "[1.0, 1.0, 1.0]"
    HIDDEN_SIZE: int = 64
    NUM_LAYERS: int = 2

    model_config = {"env_file": ".env"}
    
    @property
    def mean_array(self) -> np.ndarray:
        if isinstance(self.MEAN, str):
            # Парсим строку как список
            try:
                mean_list = ast.literal_eval(self.MEAN)
            except:
                mean_list = [0.0, 0.0, 0.0]
        else:
            mean_list = self.MEAN
        return np.array(mean_list, dtype=np.float32)
    
    @property
    def std_array(self) -> np.ndarray:
        if isinstance(self.STD, str):
            # Парсим строку как список
            try:
                std_list = ast.literal_eval(self.STD)
            except:
                std_list = [1.0, 1.0, 1.0]
        else:
            std_list = self.STD
        return np.array(std_list, dtype=np.float32)

settings = Settings()
