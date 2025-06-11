import requests
import json

# Тестовые данные в новом формате (x, y, t)
test_data = {
    "points": [
        {"x": 0, "y": 0, "t": 0},
        {"x": 1, "y": 1, "t": 1},
        {"x": 4, "y": 4, "t": 4},
        {"x": 6, "y": 6, "t": 6},
        {"x": 7, "y": 7, "t": 7}
    ]
}

# URL сервера
url = "http://localhost:8001/predict/"

try:
    # Отправляем POST запрос
    response = requests.post(url, json=test_data)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nПредсказанная следующая точка:")
        print(f"x: {result['x']}")
        print(f"y: {result['y']}")
        print(f"t: {result['t']}")
    else:
        print(f"Ошибка: {response.text}")
        
except Exception as e:
    print(f"Ошибка при отправке запроса: {e}")
