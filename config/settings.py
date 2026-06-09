import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "SAR Wind Gujarat API"
    VERSION: str = "1.0.0"

    GEE_PROJECT_ID: str = os.getenv("GEE_PROJECT_ID", "vocal-raceway-470111-k8")

    MODEL_WEIGHTS_PATH: str = "model/weights/m64rn4_weights.pth"

    SAR_SCALE_METERS: int = 500
    TILE_SIZE: int = 20
    CHANNELS: int = 64
    NUM_BLOCKS: int = 4

    model_config = SettingsConfigDict(
        env_file = ".env", 
        env_file_encoding = 'utf-8',
        extra = 'ignore'
    )

settings = Settings()
