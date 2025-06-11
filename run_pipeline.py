#!/usr/bin/env python3
"""
Скрипт для полного запуска пайплайна обучения и тестирования модели
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Выполнение команды с логированием"""
    print(f"\n{'='*50}")
    print(f"Выполняется: {description}")
    print(f"Команда: {command}")
    print(f"{'='*50}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"❌ Ошибка при выполнении: {description}")
        return False
    else:
        print(f"✅ Успешно выполнено: {description}")
        return True

def setup_directories():
    """Создание необходимых директорий"""
    dirs = [
        "data/processed",
        "training/checkpoints",
        "logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Создана директория: {dir_path}")

def main():
    parser = argparse.ArgumentParser(description="Пайплайн обучения модели предсказания полета БПЛА")
    parser.add_argument("--skip-data", action="store_true", help="Пропустить генерацию данных")
    parser.add_argument("--skip-training", action="store_true", help="Пропустить обучение")
    parser.add_argument("--skip-evaluation", action="store_true", help="Пропустить оценку")
    parser.add_argument("--skip-tests", action="store_true", help="Пропустить тесты")
    
    args = parser.parse_args()
    
    print("🚁 Запуск пайплайна обучения модели предсказания полета БПЛА")
    
    # Создание директорий
    setup_directories()
    
    # 1. Предобработка и генерация данных
    if not args.skip_data:
        if not run_command("python data/preprocess.py", "Предобработка данных"):
            print("❌ Ошибка при предобработке данных")
            return 1
    
    # 2. Обучение модели
    if not args.skip_training:
        if not run_command("python training/train.py", "Обучение модели"):
            print("❌ Ошибка при обучении модели")
            return 1
    
    # 3. Оценка модели
    if not args.skip_evaluation:
        if not run_command("python scripts/evaluate.py", "Оценка модели"):
            print("⚠️ Предупреждение: не удалось выполнить оценку модели")
    
    # 4. Запуск тестов
    if not args.skip_tests:
        if not run_command("python tests/test_model.py", "Тестирование модели"):
            print("⚠️ Предупреждение: тесты модели не прошли")
        
        if not run_command("python tests/test_predict.py", "Тестирование API"):
            print("⚠️ Предупреждение: тесты API не прошли")
    
    print("\n🎉 Пайплайн завершен!")
    print("\nДля запуска сервиса используйте:")
    print("uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    print("\nИли с помощью Docker:")
    print("docker-compose up --build")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
