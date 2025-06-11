# Makefile для управления проектом предсказания полета БПЛА

.PHONY: help install data train evaluate test serve clean docker-build docker-run

help: ## Показать справку
	@echo "Доступные команды:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Установить зависимости
	pip install -r requirements.txt

data: ## Сгенерировать и предобработать данные
	python data/preprocess.py

train: ## Обучить модель
	python training/train.py

evaluate: ## Оценить качество модели
	python scripts/evaluate.py

test: ## Запустить тесты
	python tests/test_model.py
	python tests/test_predict.py

test-pytest: ## Запустить тесты через pytest
	pytest tests/ -v

serve: ## Запустить сервис локально
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pipeline: ## Запустить полный пайплайн
	python run_pipeline.py

clean: ## Очистить временные файлы
	@echo "Очистка временных файлов..."
	if exist __pycache__ rmdir /s /q __pycache__
	if exist .pytest_cache rmdir /s /q .pytest_cache
	if exist *.egg-info rmdir /s /q *.egg-info
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

docker-build: ## Собрать Docker образ
	docker-compose build

docker-run: ## Запустить в Docker
	docker-compose up

docker-dev: ## Запустить в Docker с перезагрузкой
	docker-compose up --build

format: ## Форматировать код
	black app/ training/ data/ scripts/ tests/
	isort app/ training/ data/ scripts/ tests/

lint: ## Проверить код
	flake8 app/ training/ data/ scripts/ tests/
	pylint app/ training/ data/ scripts/ tests/

setup-dev: install ## Настроить окружение разработки
	pip install black isort flake8 pylint
	@echo "Окружение разработки настроено!"

all: data train evaluate test ## Выполнить все этапы
