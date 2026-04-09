import os

class Settings:
    app_name:    str = "Corelix API"
    app_version: str = "1.0.0"
    api_prefix:  str = "/v1"

    @property
    def app_env(self) -> str:
        return os.environ.get("APP_ENV", "development")

    @property
    def allowed_origins(self) -> list[str]:
        raw = os.environ.get("ALLOWED_ORIGINS", "*")
        if raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

settings = Settings()
