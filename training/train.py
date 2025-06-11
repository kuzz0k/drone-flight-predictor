import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from app.models.network import LSTMNetwork
from training.dataset import FlightDataset, create_sequences
from app.core.utils import normalize, denormalize
from app.core.config import settings

def load_config(config_path: str = "training/config.yaml") -> dict:
    """Загрузка конфигурации обучения"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Обучение на одной эпохе"""
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        # outputs shape: (batch_size, 1, 3), batch_y shape: (batch_size, 3)
        outputs = outputs.squeeze(1)
        
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    """Валидация на одной эпохе"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Денормализация для расчета метрик в исходном масштабе
    pred_denorm = denormalize(predictions)
    target_denorm = denormalize(targets)
    
    mse = mean_squared_error(target_denorm, pred_denorm)
    mae = mean_absolute_error(target_denorm, pred_denorm)
    
    return total_loss / len(dataloader), mse, mae

def plot_training_history(train_losses, val_losses, save_path="training/training_history.png"):
    """Построение графика истории обучения"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """Основная функция обучения"""
    # Загружаем конфигурацию
    config = load_config()
    
    # Устройство для обучения
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Создаем директории
    os.makedirs("training/checkpoints", exist_ok=True)
    
    # Загружаем данные
    print("Загрузка данных...")
    df = pd.read_csv("data/processed/flight_data.csv")
    
    # Создаем последовательности
    print("Создание последовательностей...")
    X, y = create_sequences(df, window_size=5)
    
    # Нормализация данных
    print("Нормализация данных...")
    X_norm = normalize(X)
    y_norm = normalize(y)
    
    # Создаем dataset
    dataset = FlightDataset(X_norm, y_norm)
    
    # Разделяем на train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Создаем dataloader'ы
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"Размер обучающей выборки: {len(train_dataset)}")
    print(f"Размер валидационной выборки: {len(val_dataset)}")
      # Создаем модель
    model = LSTMNetwork(
        input_size=3,
        hidden_size=settings.HIDDEN_SIZE,
        num_layers=settings.NUM_LAYERS
    ).to(device)
    
    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    # Обучение
    print("Начало обучения...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        # Обучение
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Валидация
        val_loss, mse, mae = validate_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "training/checkpoints/best.pt")
            print(f"  Сохранена лучшая модель (val_loss: {val_loss:.6f})")
        
        print()
    
    # Сохраняем график обучения
    plot_training_history(train_losses, val_losses)
    
    print("Обучение завершено!")
    print(f"Лучшая валидационная потеря: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()
