from ast import literal_eval
from functools import lru_cache

from minio import Minio
from src.config import Config

cfg = Config()

MINIO_CLIENT = Minio(f"{cfg.MINIO_ENDPOINT}:{cfg.MINIO_PORT}",
                     access_key=cfg.MINIO_ACCESS_KEY, secret_key=cfg.MINIO_SECRET_KEY, secure=literal_eval(cfg.MINIO_SECURE))


@lru_cache()
def MINIO_CLIENT_GETTER():
    from minio import Minio

    return Minio(f"{cfg.MINIO_ENDPOINT}:{cfg.MINIO_PORT}", access_key=cfg.MINIO_ACCESS_KEY,
                 secret_key=cfg.MINIO_SECRET_KEY, secure=literal_eval(cfg.MINIO_SECURE))


@lru_cache()
def DB_ENGINE_GETTER():
    from sqlalchemy import create_engine

    return create_engine(f"postgresql+psycopg2://{cfg.DB_USERNAME}:{cfg.DB_PASSWORD}@{cfg.DB_HOSTNAME}:{cfg.DB_PORT}/{cfg.DB_NAME}")


