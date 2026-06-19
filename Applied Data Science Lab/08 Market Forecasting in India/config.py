from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    alpha_api_key: str
    db_name: str = "stock_data.db"
    model_directory: str = "models"

    class Config:
        env_file = ".env"


settings = Settings()