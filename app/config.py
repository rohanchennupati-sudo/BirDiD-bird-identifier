from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    environment: str = "development"
    log_level: str = "INFO"

    max_image_size_mb: int = 10
    rate_limit_per_minute: int = 20

    allowed_origins: str = (
        "http://localhost:3000,http://localhost:8000"
    )

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()