import configparser
import logging


class Config:

    def __init__(self):
        self._config = configparser.ConfigParser()

    def init_from(self, config_file):
        status = self._config.read(config_file)
        return len(status) != 0

    def get_log_level(self):
        log_level = self._config["LOGGING"]["log_level"]
        if log_level == "CRITICAL":
            return logging.CRITICAL
        elif log_level == "ERROR":
            return logging.ERROR
        elif log_level == "WARNING":
            return logging.WARNING
        elif log_level == "INFO":
            return logging.INFO
        elif log_level == "DEBUG":
            return logging.DEBUG
        elif log_level == "NOTSET":
            return logging.NOTSET
        raise TypeError(f"Unexpected LOGGING.log_level value: {log_level}")
