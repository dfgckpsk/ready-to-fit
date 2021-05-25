import logging
import sys
from ._ConcatLogsHandler import _ConcatLogsHandler


class _WhiteFormatter(logging.Formatter):
    """Logging Formatter to add count warning / errors"""

    format = "%(asctime)23s %(levelname)8s %(class)20s: %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: format,
        logging.INFO: format,
        logging.WARNING: format,
        logging.ERROR: format,
        logging.CRITICAL: format
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class _ColoredFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    green = "\x1b[32m"
    grey = "\x1b[38m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)23s %(levelname)8s %(class)20s: %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def logged(original_class):
    """Decorates any class to provide log methods: _debug, _info, _warning, _error and _critical"""
    orig_init = original_class.__init__

    def __init__(self, *args, **kws):
        logger = logging.getLogger("tools")
        logger_adapter = logging.LoggerAdapter(logger, {"class": original_class.__name__})
        self._debug = logger_adapter.debug
        self._info = logger_adapter.info
        self._error = logger_adapter.error
        self._critical = logger_adapter.critical
        self._warning = logger_adapter.warning
        self._logger = logger

        orig_init(self, *args, **kws)

    original_class.__init__ = __init__
    return original_class


def get_logger(point_name):
    logger = logging.getLogger("tools")
    logger_adapter = logging.LoggerAdapter(logger, {"class": point_name})
    return logger_adapter


# @functools.lru_cache
def init_logger(config):
    log_level = config.get_log_level()
    formatter = _ColoredFormatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    db_handler = _ConcatLogsHandler()
    db_handler.setLevel(logging.INFO)
    db_handler.setFormatter(formatter)

    logger = logging.getLogger("tools")
    logger.setLevel(log_level)
    logger.addHandler(stream_handler)
    logger.addHandler(db_handler)
