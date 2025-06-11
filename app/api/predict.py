from fastapi import APIRouter, HTTPException
from app.schemas.flight import SequenceIn, PointOut
from app.core.utils import normalize, denormalize
from app.models.predictor import Predictor, load_model
import numpy as np
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

try:
    model = load_model()
    logger.info("Модель успешно загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    model = None

@router.post("/", response_model=PointOut)
def predict(seq: SequenceIn):
    """Предсказание следующей точки траектории по 5 предыдущим точкам"""
    if len(seq.points) != 5:
        raise HTTPException(400, "Нужно ровно 5 точек")
    
    try:
        # Преобразуем входные данные в список точек
        points = []
        for point in seq.points:
            points.append([point.x, point.y, point.t])
        
        points = np.array(points)  # shape: (5, 3)
        
        # Кинематический подход для предсказания
        if len(points) >= 3:
            # Берем последние 3 точки для анализа ускорения
            p1, p2, p3 = points[-3], points[-2], points[-1]
            
            # Вычисляем скорости между точками
            v1 = p2 - p1  # скорость между p1 и p2
            v2 = p3 - p2  # скорость между p2 и p3
            
            # Вычисляем ускорение
            acceleration = v2 - v1
            
            # Предсказываем следующую точку с учетом ускорения
            # Формула: next = current + velocity + 0.5 * acceleration
            kinematic_prediction = p3 + v2 + 0.5 * acceleration
            
        elif len(points) >= 2:
            # Если только 2 точки, используем линейную экстраполяцию
            p1, p2 = points[-2], points[-1]
            velocity = p2 - p1
            kinematic_prediction = p2 + velocity
            
        else:
            # Если только 1 точка, просто копируем её
            kinematic_prediction = points[-1].copy()
            kinematic_prediction[2] += 1.0  # увеличиваем время на 1
        
        # Если модель загружена, попытаемся получить её предсказание
        neural_prediction = None
        if model is not None:
            try:
                # Нормализуем входные данные для нейронной сети
                arr = seq.to_numpy()  # shape: (1, 5, 3)
                normed = normalize(arr)
                
                # Получаем предсказание от модели
                pred = model.predict(normed)  # shape: (1, 3)
                denormed = denormalize(pred)
                neural_prediction = denormed[0]
                
                logger.info(f"Neural prediction: {neural_prediction}")
                
            except Exception as e:
                logger.warning(f"Ошибка нейронной сети, используем кинематику: {e}")
        
        # Используем комбинированный подход или только кинематику
        if neural_prediction is not None:
            # Комбинируем предсказания (больше веса кинематике)
            final_prediction = 0.7 * kinematic_prediction + 0.3 * neural_prediction
            logger.info(f"Combined prediction used")
        else:
            # Используем только кинематическое предсказание
            final_prediction = kinematic_prediction
            logger.info(f"Kinematic-only prediction used")
        
        logger.info(f"Input points: {points.tolist()}")
        logger.info(f"Final prediction: {final_prediction.tolist()}")
        
        # Возвращаем результат как PointOut
        return PointOut(
            x=float(final_prediction[0]),
            y=float(final_prediction[1]),
            t=float(final_prediction[2])
        )
    
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(500, f"Ошибка при предсказании: {str(e)}")

@router.get("/health")
def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }
