import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Tuple, List

class FlightDataset(Dataset):
    """Dataset для данных полета БПЛА"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: numpy array shape (n_samples, 5, 3) - последовательности из 5 точек
            y: numpy array shape (n_samples, 3) - следующие точки
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(df: pd.DataFrame, window_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Создание последовательностей для обучения из DataFrame"""
    sequences = []
    targets = []
    
    # Группируем по trajectory_id если есть
    if 'trajectory_id' in df.columns:
        for traj_id in df['trajectory_id'].unique():
            traj_data = df[df['trajectory_id'] == traj_id]
            X_traj, y_traj = _create_windows(traj_data[['x', 'y', 't']].values, window_size)
            sequences.extend(X_traj)
            targets.extend(y_traj)
    else:
        # Если нет trajectory_id, обрабатываем как одну траекторию
        X_traj, y_traj = _create_windows(df[['x', 'y', 't']].values, window_size)
        sequences.extend(X_traj)
        targets.extend(y_traj)
    
    return np.array(sequences), np.array(targets)

def _create_windows(data: np.ndarray, window_size: int) -> Tuple[List, List]:
    """Создание скользящих окон из данных одной траектории"""
    X, y = [], []
    
    for i in range(len(data) - window_size):
        # Берем window_size точек как входную последовательность
        X.append(data[i:i+window_size])
        # Следующая точка как целевая
        y.append(data[i+window_size])
    
    return X, y