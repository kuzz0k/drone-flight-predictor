import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

from app.models.predictor import load_model
from app.core.utils import normalize, denormalize
from training.dataset import create_sequences

def evaluate_model(model_path: str = "training/checkpoints/best.pt"):
    """Оценка качества модели на тестовых данных"""
    
    # Загружаем модель
    print("Загрузка модели...")
    try:
        model = load_model()
        print("Модель успешно загружена")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return
    
    # Загружаем тестовые данные
    print("Загрузка данных...")
    df = pd.read_csv("data/processed/flight_data.csv")
    
    # Создаем последовательности
    X, y = create_sequences(df, window_size=5)
    
    # Берем последние 20% данных для тестирования
    test_size = int(0.2 * len(X))
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    print(f"Размер тестовой выборки: {len(X_test)}")
    
    # Нормализация
    X_test_norm = normalize(X_test)
    y_test_norm = normalize(y_test)
    
    # Предсказания
    print("Выполнение предсказаний...")
    predictions = []
    
    for i in range(len(X_test_norm)):
        pred = model.predict(X_test_norm[i:i+1])
        predictions.append(pred.squeeze())
    
    predictions = np.array(predictions)
    
    # Денормализация для оценки
    y_test_denorm = denormalize(y_test_norm)
    pred_denorm = denormalize(predictions)
    
    # Вычисление метрик
    mse = mean_squared_error(y_test_denorm, pred_denorm)
    mae = mean_absolute_error(y_test_denorm, pred_denorm)
    rmse = np.sqrt(mse)
    
    print(f"\nМетрики качества:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
      # Метрики по каждой координате
    coords = ['x', 'y', 't']
    for i, coord in enumerate(coords):
        mse_coord = mean_squared_error(y_test_denorm[:, i], pred_denorm[:, i])
        mae_coord = mean_absolute_error(y_test_denorm[:, i], pred_denorm[:, i])
        print(f"{coord}: MSE={mse_coord:.6f}, MAE={mae_coord:.6f}")
    
    # Построение графиков
    plot_results(y_test_denorm, pred_denorm, coords)
    
    return {
        'mse': mse,
        'mae': mae, 
        'rmse': rmse,
        'predictions': pred_denorm,
        'targets': y_test_denorm
    }

def plot_results(y_true, y_pred, coords, save_path="training/evaluation_results.png"):
    """Построение графиков результатов"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, coord in enumerate(coords):
        ax = axes[i]
        
        # Scatter plot: истинные vs предсказанные значения
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=20)
        
        # Линия идеального предсказания
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_xlabel(f'Истинные значения {coord}')
        ax.set_ylabel(f'Предсказанные значения {coord}')
        ax.set_title(f'Координата {coord}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Графики сохранены в {save_path}")

def plot_trajectory_comparison(y_true, y_pred, n_examples=3):
    """Сравнение траекторий"""
    
    fig = plt.figure(figsize=(15, 5))
    
    for i in range(min(n_examples, len(y_true))):
        ax = fig.add_subplot(1, n_examples, i+1, projection='3d')
          # Истинные точки
        ax.scatter(y_true[i, 0], y_true[i, 1], 0, 
                  c='blue', s=100, label='Истинная точка', marker='o')
        
        # Предсказанные точки
        ax.scatter(y_pred[i, 0], y_pred[i, 1], 0, 
                  c='red', s=100, label='Предсказанная точка', marker='^')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Пример {i+1}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("training/trajectory_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Сравнение траекторий сохранено в training/trajectory_comparison.png")

if __name__ == "__main__":
    results = evaluate_model()
    
    if results:
        plot_trajectory_comparison(results['targets'], results['predictions'])
        print("\nОценка модели завершена!")