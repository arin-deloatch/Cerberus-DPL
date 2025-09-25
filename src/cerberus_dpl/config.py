from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    EMBEDDING_MODEL: str = "ibm-granite/granite-embedding-30m-english"
    HF_TOKEN: str = ""

    USER_AGENT: str = "Ceberus-DPL/0.1"
    TIMEOUT_S: int = 20

    SEED_MODEL_PATH: str = "artifacts/"
    SEED_N_CENTROIDS: int = 1


settings = Settings()
