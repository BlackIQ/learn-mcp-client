# Libs
from pydantic_settings import BaseSettings, SettingsConfigDict  # Pydantic Settings


# Settings
class Settings(BaseSettings):
    # Model
    MODEL: str = "qwen2.5:3b"

    model_config = SettingsConfigDict(env_file=".env")


# Run the settings
settings = Settings()
