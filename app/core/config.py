class Settings:
    app_name:    str = "Corelix API"
    app_version: str = "1.0.0"
    app_env:     str = "development"
    api_prefix:  str = "/v1"
    allowed_origins: list = ["*"]

settings = Settings()
