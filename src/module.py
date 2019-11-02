from ast import literal_eval
from functools import lru_cache

from minio import Minio
from src.configs.db_config import Config

cfg = Config()

MINIO_CLIENT = Minio(f"{cfg.MINIO_ENDPOINT}:{cfg.MINIO_PORT}",
                     access_key=cfg.MINIO_ACCESS_KEY, secret_key=cfg.MINIO_SECRET_KEY,
                     secure=literal_eval(cfg.MINIO_SECURE))


@lru_cache(maxsize=64)
def MINIO_CLIENT_GETTER():
    from minio import Minio

    return Minio(f"{cfg.MINIO_ENDPOINT}:{cfg.MINIO_PORT}", access_key=cfg.MINIO_ACCESS_KEY,
                 secret_key=cfg.MINIO_SECRET_KEY, secure=literal_eval(cfg.MINIO_SECURE))


@lru_cache(maxsize=64)
def DATACONFIG_GETTER():
    import yaml
    with open("configs/configs.yaml") as ymlfile:
        return yaml.safe_load(ymlfile)['DataConfig']
