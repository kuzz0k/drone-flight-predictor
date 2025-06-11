import pytest
import numpy as np
import torch
from app.models.network import LSTMNetwork
from app.models.predictor import Predictor

def test_lstm_network_forward():
    """Тест прямого прохода через LSTM сеть"""
    model = LSTMNetwork(input_size=3, hidden_size=64, num_layers=2)
    
    # Тестовый вход: batch_size=2, seq_len=5, features=3
    x = torch.randn(2, 5, 3)
    
    output = model(x)
    
    # Проверяем размерность выхода
    assert output.shape == (2, 1, 3), f"Expected shape (2, 1, 3), got {output.shape}"

def test_lstm_network_single_batch():
    """Тест с одним примером"""
    model = LSTMNetwork(input_size=3, hidden_size=32, num_layers=1)
    
    # Один пример: batch_size=1, seq_len=5, features=3
    x = torch.randn(1, 5, 3)
    
    output = model(x)
    
    assert output.shape == (1, 1, 3), f"Expected shape (1, 1, 3), got {output.shape}"

def test_predictor():
    """Тест класса Predictor"""
    model = LSTMNetwork(input_size=3, hidden_size=32, num_layers=1)
    predictor = Predictor(model)
    
    # Тестовые данные
    x = np.random.randn(1, 5, 3).astype(np.float32)
    
    prediction = predictor.predict(x)
    
    # Проверяем, что предсказание имеет правильную форму
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1, 1, 3), f"Expected shape (1, 1, 3), got {prediction.shape}"

def test_predict_shape():
    """Тест размерности предсказания"""
    model = LSTMNetwork(input_size=3, hidden_size=16, num_layers=1)
    predictor = Predictor(model)
    
    dummy = np.zeros((1, 5, 3), dtype=np.float32)
    pred = predictor.predict(dummy)
    assert pred.shape == (1, 1, 3), f"Expected shape (1, 1, 3), got {pred.shape}"

def test_lstm_deterministic():
    """Тест детерминированности модели"""
    torch.manual_seed(42)
    model = LSTMNetwork(input_size=3, hidden_size=16, num_layers=1)
    
    x = torch.randn(1, 5, 3)
    
    # Два прохода с одними и теми же данными
    output1 = model(x)
    output2 = model(x)
    
    # Результаты должны быть одинаковыми
    assert torch.allclose(output1, output2), "Model should be deterministic"

if __name__ == "__main__":
    test_lstm_network_forward()
    test_lstm_network_single_batch()
    test_predictor()
    test_predict_shape()
    test_lstm_deterministic()
    print("Все тесты модели прошли успешно!")
