import numpy as np
from app.core.config import settings

def normalize(arr: np.ndarray) -> np.ndarray:
    """Нормализация массива (batch_size, seq_len, features) или (seq_len, features)"""
    mean = settings.mean_array
    std = settings.std_array
    return (arr - mean) / std

def denormalize(arr: np.ndarray) -> np.ndarray:
    """Денормализация массива"""
    mean = settings.mean_array
    std = settings.std_array
    return arr * std + mean
