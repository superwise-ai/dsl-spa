from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BEARER_TOKEN: str = ""
    SUPERWISE_CLIENT_ID: str = ""
    SUPERWISE_CLIENT_SECRET: str = ""

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = True