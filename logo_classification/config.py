from pydantic import BaseSettings


class Settings(BaseSettings):
    DATASET_FOLDER: str

    class Config:
        env_file: str = ".env"


settings = Settings()
