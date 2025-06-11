import requests
import json

# Тестовые данные
test_data = {
    "points": [
        {"x": 0.0, "y": 0.0, "z": 10.0, "t": 0.0},
        {"x": 1.0, "y": 1.0, "z": 10.5, "t": 1.0},
        {"x": 2.0, "y": 2.0, "z": 11.0, "t": 2.0},
        {"x": 3.0, "y": 3.0, "z": 11.5, "t": 3.0},
        {"x": 4.0, "y": 4.0, "z": 12.0, "t": 4.0}
    ]
}

print("Тестирование API...")

# Проверяем health endpoint
print("\n1. Проверка health endpoint:")
response = requests.get("http://localhost:8000/predict/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Тестируем predict endpoint
print("\n2. Тестирование predict endpoint:")
response = requests.post("http://localhost:8000/predict/", json=test_data)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print(f"Предсказанная точка:")
    print(f"  x: {result['x']:.6f}")
    print(f"  y: {result['y']:.6f}") 
    print(f"  z: {result['z']:.6f}")
    print(f"  t: {result['t']:.6f}")
    print(f"\nПоследняя входная точка была: x=4.0, y=4.0, z=12.0, t=4.0")
    print(f"Предсказанная точка: x={result['x']:.1f}, y={result['y']:.1f}, z={result['z']:.1f}, t={result['t']:.1f}")
else:
    print(f"Ошибка: {response.text}")

# Тест с неправильным количеством точек
print("\n3. Тест с неправильным количеством точек:")
bad_data = {"points": [{"x": 0.0, "y": 0.0, "z": 0.0, "t": 0.0}]}
response = requests.post("http://localhost:8000/predict/", json=bad_data)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

print("\nТестирование завершено!")
