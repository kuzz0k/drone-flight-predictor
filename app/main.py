from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys

from app.api.predict import router as predict_router
from app.core.config import settings

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Микросервис для предсказания полета БПЛА с использованием LSTM",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/predict", tags=["predict"])

@app.on_event("startup")
async def startup_event():
    logger.info(f"Запуск {settings.PROJECT_NAME} v{settings.VERSION}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Остановка сервиса")

@app.get("/")
async def root():
    return {
        "message": "Drone Flight Predictor API",
        "version": settings.VERSION,
        "docs": "/docs"
    }
