import logging.config
from datetime import datetime as dt
from typing import Dict
from pythonjsonlogger.jsonlogger import JsonFormatter


def setup_logging(config: Dict) -> None:
    """
    Setup the service logging schema.

    :param config: log settings
    :return:
    """

    logging.config.dictConfig(config)


def get_logger_settings(log_level):
    settings = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"global": {"()": "logs.GlobalFilter"}},
        "formatters": {
            "json": {
                "format": "[%(ts)s %(level)s %(message)s %(category)s]",
                "class": "logs.CustomJsonFormatter",
            }
        },
        "handlers": {
            "json": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "filters": ["global"],
            }
        },
        "loggers": {"": {"handlers": ["json"], "level": log_level}},
    }
    return settings


def timestamp_filter() -> str:  # pragma: no cover
    now = dt.utcnow()
    DT_FMT = "%Y-%m-%dT%H:%M:%S%Z"
    return now.strftime(DT_FMT)


class GlobalFilter(logging.Filter):  # pragma: no cover
    def filter(self, record):
        record.ts = timestamp_filter()
        record.level = record.levelname
        record.category = record.name
        return True


class CustomJsonFormatter(JsonFormatter):  # pragma: no cover
    def process_log_record(self, log_record):
        if "exc_info" in log_record:
            log_record["stack"] = log_record.pop("exc_info")
        return log_record
