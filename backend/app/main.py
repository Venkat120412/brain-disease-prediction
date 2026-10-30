"""FastAPI application entry point."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api import predict, train

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
log = logging.getLogger("brain_disease.main")

app = FastAPI(
    title="Brain Disease Prediction System",
    version="1.0.0",
    description="Predict Stroke, Alzheimer's, Parkinson's, and Brain Tumor risk from patient data.",
    docs_url="/docs", redoc_url="/redoc",
)

app.add_middleware(CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(predict.router, prefix="/api", tags=["Prediction"])
app.include_router(train.router,   prefix="/api", tags=["Training"])

@app.on_event("startup")
async def startup():
    from app.services.ml_engine import disease_ready
    from app.core.config import DISEASES, DISEASE_LABELS
    log.info("=" * 55)
    log.info("Brain Disease Prediction System  v1.0")
    for d in DISEASES:
        state = "✓ Ready" if disease_ready(d) else "✗ Not trained"
        log.info("  %-20s %s", DISEASE_LABELS[d], state)
    log.info("Docs: http://localhost:%d/docs", settings.backend_port)
    log.info("=" * 55)

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Brain Disease Prediction API", "docs": "/docs", "health": "/api/health"}
