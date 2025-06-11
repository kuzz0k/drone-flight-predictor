#!/usr/bin/env python3
"""
Финальный тест для API предсказания траектории дрона
Тестирует различные сценарии с данными в формате {x, y, t}
"""

import requests
import json

def test_prediction_api():
    """Тестирует API предсказания траектории"""
    
    base_url = "http://localhost:8001"
    
    # Проверяем health endpoint
    print("🔍 Проверка состояния сервера...")
    health_response = requests.get(f"{base_url}/predict/health")
    if health_response.status_code == 200:
        health_data = health_response.json()
        print(f"✅ Сервер работает: {health_data}")
    else:
        print(f"❌ Сервер недоступен: {health_response.status_code}")
        return
    
    # Тестовые сценарии
    test_cases = [
        {
            "name": "Линейная траектория",
            "description": "Точки движутся по прямой с постоянной скоростью",
            "points": [
                {"x": 0.0, "y": 0.0, "t": 0.0},
                {"x": 1.0, "y": 1.0, "t": 1.0},
                {"x": 2.0, "y": 2.0, "t": 2.0},
                {"x": 3.0, "y": 3.0, "t": 3.0},
                {"x": 4.0, "y": 4.0, "t": 4.0}
            ],
            "expected": {"x": 5.0, "y": 5.0, "t": 5.0}
        },
        {
            "name": "Ускоряющаяся траектория",
            "description": "Точки движутся с ускорением",
            "points": [
                {"x": 0.0, "y": 0.0, "t": 0.0},
                {"x": 1.0, "y": 1.0, "t": 1.0},
                {"x": 3.0, "y": 4.0, "t": 2.0},
                {"x": 6.0, "y": 9.0, "t": 3.0},
                {"x": 10.0, "y": 16.0, "t": 4.0}
            ],
            "expected": {"x": 15.0, "y": 25.0, "t": 5.0}
        },
        {
            "name": "Исходные тестовые данные",
            "description": "Данные из вашего примера",
            "points": [
                {"x": 0, "y": 0, "t": 0},
                {"x": 1, "y": 1, "t": 1},
                {"x": 4, "y": 4, "t": 4},
                {"x": 6, "y": 6, "t": 6},
                {"x": 7, "y": 7, "t": 7}
            ],
            "expected": {"x": 8.0, "y": 8.0, "t": 8.0}
        },
        {
            "name": "Движение только по X",
            "description": "Движение только вдоль оси X",
            "points": [
                {"x": 0.0, "y": 5.0, "t": 0.0},
                {"x": 2.0, "y": 5.0, "t": 1.0},
                {"x": 4.0, "y": 5.0, "t": 2.0},
                {"x": 6.0, "y": 5.0, "t": 3.0},
                {"x": 8.0, "y": 5.0, "t": 4.0}
            ],
            "expected": {"x": 10.0, "y": 5.0, "t": 5.0}
        }
    ]
    
    print("\n🚀 Тестирование API предсказания...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Тест {i}: {test_case['name']}")
        print(f"Описание: {test_case['description']}")
        
        # Отправляем запрос
        payload = {"points": test_case["points"]}
        response = requests.post(f"{base_url}/predict/", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            expected = test_case["expected"]
            
            print(f"  📍 Ожидаемый результат: x={expected['x']:.2f}, y={expected['y']:.2f}, t={expected['t']:.2f}")
            print(f"  🎯 Полученный результат: x={result['x']:.2f}, y={result['y']:.2f}, t={result['t']:.2f}")
            
            # Проверяем точность (допуск ±0.5)
            tolerance = 0.5
            x_ok = abs(result['x'] - expected['x']) <= tolerance
            y_ok = abs(result['y'] - expected['y']) <= tolerance
            t_ok = abs(result['t'] - expected['t']) <= tolerance
            
            if x_ok and y_ok and t_ok:
                print(f"  ✅ ПРОЙДЕН")
            else:
                print(f"  ⚠️  ОТКЛОНЕНИЕ от ожидаемого")
        else:
            print(f"  ❌ ОШИБКА: {response.status_code} - {response.text}")
        
        print()
    
    print("🏁 Тестирование завершено!")

if __name__ == "__main__":
    test_prediction_api()
