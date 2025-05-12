from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BEARER_TOKEN: str = ""
    # Add any custom settings here

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = True