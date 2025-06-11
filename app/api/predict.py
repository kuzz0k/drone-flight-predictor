from fastapi import APIRouter, HTTPException
from app.schemas.flight import SequenceIn, PointOut
from app.core.utils import normalize, denormalize
from app.models.predictor import Predictor, load_model
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Загружаем модель при старте сервиса
try:
    model = load_model()
    logger.info("Модель успешно загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    model = None

@router.post("/", response_model=PointOut)
def predict(seq: SequenceIn):
    """Предсказание следующей точки траектории по 5 предыдущим точкам"""
    if model is None:
        raise HTTPException(500, "Модель не загружена")
    
    if len(seq.points) != 5:
        raise HTTPException(400, "Нужно ровно 5 точек")
    
    try:
        # Преобразуем в numpy array
        arr = seq.to_numpy()  # shape: (1, 5, 4)
        
        # Нормализуем
        normed = normalize(arr)
        
        # Предсказываем
        pred = model.predict(normed)  # shape: (1, 1, 4)
        
        # Денормализуем
        pred_squeezed = pred.squeeze()  # shape: (4,)
        denormed = denormalize(pred_squeezed.reshape(1, -1))[0]  # shape: (4,)
        
        return PointOut.from_array(denormed)
    
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
