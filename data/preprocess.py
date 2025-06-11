import pandas as pd
import numpy as np
from typing import Tuple, List
import os
from pathlib import Path

def load_flight_data(file_path: str) -> pd.DataFrame:
    """Загрузка данных полета БПЛА из CSV файла"""
    df = pd.read_csv(file_path)
      # Ожидаем колонки: x, y, t (время)
    required_cols = ['x', 'y', 't']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV файл должен содержать колонки: {required_cols}")
    
    # Сортируем по времени
    df = df.sort_values('t').reset_index(drop=True)
    
    return df[required_cols]

def generate_synthetic_flight_data(n_trajectories: int = 100, 
                                  trajectory_length: int = 50) -> pd.DataFrame:
    """Генерация синтетических данных полета БПЛА"""
    all_data = []
    
    for traj_id in range(n_trajectories):
        # Генерируем случайную траекторию
        t = np.linspace(0, 10, trajectory_length)
          # Создаем плавные траектории с некоторой случайностью
        x = np.sin(t * 0.5 + np.random.random()) * 10 + np.random.normal(0, 0.1, len(t))
        y = np.cos(t * 0.3 + np.random.random()) * 8 + np.random.normal(0, 0.1, len(t))
        
        traj_data = pd.DataFrame({
            'x': x,
            'y': y, 
            't': t,
            'trajectory_id': traj_id
        })
        
        all_data.append(traj_data)
    
    return pd.concat(all_data, ignore_index=True)

def calculate_normalization_stats(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Вычисление статистик для нормализации"""
    features = df[['x', 'y', 't']].values
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    
    # Избегаем деления на ноль
    std = np.where(std == 0, 1.0, std)
    
    return mean, std

def save_normalization_stats(mean: np.ndarray, std: np.ndarray, 
                           output_path: str = "configs/development.env"):
    """Сохранение статистик нормализации в .env файл"""
    mean_str = ",".join(map(str, mean))
    std_str = ",".join(map(str, std))
    
    env_content = f"""MODEL_PATH=training/checkpoints/best.pt
MEAN=[{mean_str}]
STD=[{std_str}]
DEVICE=cpu
"""
    
    with open(output_path, 'w') as f:
        f.write(env_content)
    
    print(f"Статистики нормализации сохранены в {output_path}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")

def main():
    """Основная функция предобработки данных"""
    # Создаем директории если их нет
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("training/checkpoints", exist_ok=True)
    
    # Генерируем синтетические данные (замените на загрузку реальных данных)
    print("Генерация синтетических данных...")
    df = generate_synthetic_flight_data(n_trajectories=200, trajectory_length=100)
    
    # Сохраняем сырые данные
    df.to_csv("data/processed/flight_data.csv", index=False)
    print(f"Сохранено {len(df)} точек в data/processed/flight_data.csv")
    
    # Вычисляем статистики нормализации
    mean, std = calculate_normalization_stats(df)
    
    # Сохраняем статистики
    save_normalization_stats(mean, std)
    
    print("Предобработка данных завершена!")

if __name__ == "__main__":
    main()