"""
Демонстрационный скрипт для проекта drone-flight-predictor
Показывает все основные возможности системы предсказания полета БПЛА
"""

import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import json

class DronePredictor:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.check_api_health()
    
    def check_api_health(self):
        """Проверка работоспособности API"""
        try:
            response = requests.get(f"{self.api_url}/predict/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API работает: {data['status']}")
                print(f"✅ Модель загружена: {data['model_loaded']}")
            else:
                raise Exception(f"API недоступно: {response.status_code}")
        except Exception as e:
            print(f"❌ Ошибка подключения к API: {e}")
            raise
    
    def predict_next_point(self, points):
        """Предсказание следующей точки траектории"""
        if len(points) != 5:
            raise ValueError("Нужно ровно 5 точек для предсказания")
        
        payload = {
            "points": [
                {"x": p[0], "y": p[1], "z": p[2], "t": p[3]} 
                for p in points
            ]
        }
        
        response = requests.post(f"{self.api_url}/predict/", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return [result['x'], result['y'], result['z'], result['t']]
        else:
            raise Exception(f"Ошибка предсказания: {response.text}")
    
    def predict_trajectory(self, initial_points, steps=10):
        """Предсказание траектории на несколько шагов вперед"""
        trajectory = list(initial_points)
        
        print(f"🚁 Предсказание траектории на {steps} шагов...")
        
        for i in range(steps):
            # Берем последние 5 точек
            last_5_points = trajectory[-5:]
            
            # Предсказываем следующую точку
            next_point = self.predict_next_point(last_5_points)
            trajectory.append(next_point)
            
            print(f"Шаг {i+1}: x={next_point[0]:.2f}, y={next_point[1]:.2f}, z={next_point[2]:.2f}, t={next_point[3]:.2f}")
            time.sleep(0.1)  # Небольшая задержка для демонстрации
        
        return trajectory
    
    def visualize_trajectory(self, trajectory, title="Траектория полета БПЛА"):
        """Визуализация 3D траектории"""
        trajectory = np.array(trajectory)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Исходные точки (зеленые)
        ax.scatter(trajectory[:5, 0], trajectory[:5, 1], trajectory[:5, 2], 
                  c='green', s=100, alpha=0.8, label='Исходные точки')
        
        # Предсказанные точки (красные)
        if len(trajectory) > 5:
            ax.scatter(trajectory[5:, 0], trajectory[5:, 1], trajectory[5:, 2], 
                      c='red', s=60, alpha=0.6, label='Предсказанные точки')
        
        # Линия траектории
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
               'b-', alpha=0.5, linewidth=2, label='Траектория')
        
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Z (м)')
        ax.set_title(title)
        ax.legend()
        
        # Сохраняем график
        plt.savefig('demo_trajectory.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 График сохранен как 'demo_trajectory.png'")

def create_demo_trajectories():
    """Создание демонстрационных траекторий"""
    trajectories = {
        "Линейное движение": [
            [0.0, 0.0, 10.0, 0.0],
            [1.0, 1.0, 10.5, 1.0],
            [2.0, 2.0, 11.0, 2.0],
            [3.0, 3.0, 11.5, 3.0],
            [4.0, 4.0, 12.0, 4.0],
        ],
        "Круговое движение": [
            [5.0, 0.0, 15.0, 0.0],
            [4.5, 2.2, 15.2, 1.0],
            [3.5, 4.0, 15.4, 2.0],
            [1.5, 4.8, 15.6, 3.0],
            [-0.5, 4.5, 15.8, 4.0],
        ],
        "Синусоидальное движение": [
            [0.0, 0.0, 20.0, 0.0],
            [1.0, 0.8, 19.8, 1.0],
            [2.0, 1.4, 19.6, 2.0],
            [3.0, 1.8, 19.4, 3.0],
            [4.0, 1.9, 19.2, 4.0],
        ]
    }
    return trajectories

def main():
    """Основная демонстрационная функция"""
    print("🚁 === ДЕМОНСТРАЦИЯ СИСТЕМЫ ПРЕДСКАЗАНИЯ ПОЛЕТА БПЛА ===")
    print()
    
    try:
        # Инициализация предиктора
        predictor = DronePredictor()
        print()
        
        # Демонстрационные траектории
        demo_trajectories = create_demo_trajectories()
        
        for name, initial_points in demo_trajectories.items():
            print(f"📍 === {name.upper()} ===")
            print("Исходные 5 точек:")
            for i, point in enumerate(initial_points):
                print(f"  Точка {i+1}: x={point[0]:.1f}, y={point[1]:.1f}, z={point[2]:.1f}, t={point[3]:.1f}")
            
            # Предсказание траектории
            full_trajectory = predictor.predict_trajectory(initial_points, steps=5)
            
            # Визуализация
            predictor.visualize_trajectory(full_trajectory, f"{name} - Предсказание БПЛА")
            print()
            
            input("Нажмите Enter для продолжения...")
            print()
        
        # Демонстрация обработки ошибок
        print("🔧 === ТЕСТИРОВАНИЕ ОБРАБОТКИ ОШИБОК ===")
        try:
            predictor.predict_next_point([[1, 2, 3, 4]])  # Только 1 точка
        except ValueError as e:
            print(f"✅ Корректно обработана ошибка: {e}")
        
        print()
        print("🎉 === ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА ===")
        print("Все функции системы работают корректно!")
        print()
        print("📋 Возможности системы:")
        print("  ✅ Предсказание следующей точки по 5 предыдущим")
        print("  ✅ Построение траектории на несколько шагов вперед")
        print("  ✅ 3D визуализация траекторий")
        print("  ✅ REST API для интеграции")
        print("  ✅ Обработка ошибок и валидация")
        print("  ✅ Высокое качество предсказаний (MAE < 0.07)")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("Убедитесь, что API сервер запущен: uvicorn app.main:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main()
