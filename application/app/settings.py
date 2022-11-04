from pydantic import BaseSettings


class Settings(BaseSettings):
    hostname: str = 'localhost'
    port: int = 9999
    project_name: str = 'Whoosh'
    ssl_keyfile: str = ''
    ssl_certfile: str = ''

    class Config:
        case_sensitive = False
        env_file = '.env'
        env_file_encoding = 'utf-8'

