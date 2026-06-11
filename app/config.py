from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.
    Pydantic validates types at startup — bad config fails fast.
    """
    environment: str = "development"
    log_level: str = "INFO"
    max_image_size_mb: int = 10
    rate_limit_per_minute: int = 20
    allowed_origins_list: str = "http://localhost:3000,http://localhost:8000"
     # Convert comma-separated string to list
    # Path to trained model checkpoint (empty until training is done)
    model_checkpoint_path: str = "models/best_model_finetuned.pth"
    class_names_path: str = "models/class_names.json"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

@lru_cache
def get_settings() -> Settings:
    return Settings()
