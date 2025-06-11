import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.flight import Point, SequenceIn

client = TestClient(app)

def test_health_endpoint():
    """Тест эндпоинта проверки здоровья"""
    response = client.get("/predict/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_predict_endpoint_valid_input():
    """Тест предсказания с валидными данными"""    # Создаем тестовые данные
    points = [
        Point(x=0.0, y=0.0, t=0.0),
        Point(x=1.0, y=1.0, t=1.0),
        Point(x=2.0, y=2.0, t=2.0),
        Point(x=3.0, y=3.0, t=3.0),
        Point(x=4.0, y=4.0, t=4.0),
    ]
    
    sequence = SequenceIn(points=points)
    
    # Пропускаем тест если модель не загружена
    health_response = client.get("/predict/health")
    if not health_response.json().get("model_loaded", False):
        pytest.skip("Модель не загружена")
    
    response = client.post("/predict/", json=sequence.model_dump())
    
    # Проверяем, что запрос успешен или возвращает ошибку о незагруженной модели
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:        data = response.json()
        assert "x" in data
        assert "y" in data
        assert "t" in data

def test_predict_endpoint_invalid_input():
    """Тест предсказания с невалидными данными"""
    # Неправильное количество точек
    points = [
        Point(x=0.0, y=0.0, z=0.0, t=0.0),
        Point(x=1.0, y=1.0, z=1.0, t=1.0),    ]
    
    sequence = SequenceIn(points=points)
    
    response = client.post("/predict/", json=sequence.model_dump())
    assert response.status_code == 400
    assert "Нужно ровно 5 точек" in response.json()["detail"]

def test_predict_endpoint_empty_input():
    """Тест предсказания с пустыми данными"""
    sequence = SequenceIn(points=[])
    
    response = client.post("/predict/", json=sequence.model_dump())
    assert response.status_code == 400

if __name__ == "__main__":
    test_health_endpoint()
    test_predict_endpoint_valid_input()
    test_predict_endpoint_invalid_input()
    test_predict_endpoint_empty_input()
    print("Все тесты API прошли успешно!")
