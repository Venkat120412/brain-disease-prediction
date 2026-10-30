"""Central application configuration."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR  = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        case_sensitive=False,
        extra="ignore",
    )
    backend_host: str   = "0.0.0.0"
    backend_port: int   = 8000
    debug: bool         = True
    cors_origins: str   = "http://localhost:5173,http://localhost:3000"
    max_upload_size_mb: int = 50
    random_seed: int    = 42
    test_size: float    = 0.20

    data_dir:       Path = PROJECT_ROOT / "data"
    artifacts_dir:  Path = PROJECT_ROOT / "data" / "artifacts"
    models_dir:     Path = PROJECT_ROOT / "data" / "artifacts" / "models"
    raw_data_dir:   Path = PROJECT_ROOT / "data" / "raw"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_dirs(self):
        for d in [self.data_dir, self.artifacts_dir, self.models_dir, self.raw_data_dir]:
            d.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_dirs()

# Disease registry — add new diseases here
DISEASES = ["stroke", "alzheimers", "parkinsons", "brain_tumor"]

DISEASE_LABELS = {
    "stroke":       "Stroke",
    "alzheimers":   "Alzheimer's Disease",
    "parkinsons":   "Parkinson's Disease",
    "brain_tumor":  "Brain Tumor",
}
