from contextlib import asynccontextmanager
import structlog
import uuid
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routers import health, predict
logger = structlog.get_logger(__name__)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("server_starting")
    yield
    logger.info("server_stopping")
app = FastAPI(
    title="Avian Intelligence API",
    description="Bird species identifier — 200 species, trained EfficientNet-B3",
    version="1.0.0",
    lifespan=lifespan,)
# Add request ID to every request for tracing
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())[:8]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response
app.include_router(health.router)
app.include_router(predict.router, prefix="/api/v1")
# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
async def root():
    return FileResponse("static/index.html")