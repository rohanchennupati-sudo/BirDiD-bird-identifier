from contextlib import asynccontextmanager
import uuid

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

from app.config import get_settings
from app.middleware.rate_limiter import limiter
from app.routers import health, predict
from app.services.pytorch_classifier import PyTorchBirdClassifier

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trained model once, so requests don't reload it from disk.
    settings = get_settings()
    classifier = PyTorchBirdClassifier(settings.model_checkpoint_path)
    if classifier.is_available:
        classifier.load()
        app.state.classifier = classifier
    else:
        app.state.classifier = None
    logger.info("server_starting", model_loaded=app.state.classifier is not None)
    yield
    logger.info("server_stopping")


app = FastAPI(
    title="BirDiD API",
    description="Bird species identifier — 200 species, EfficientNet-B3",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())[:8]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


app.include_router(health.router)
app.include_router(predict.router, prefix="/api/v1")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")
