from .logging import _WhiteFormatter, logged, init_logger
import logging
from .config import Config
import os

_config = Config()
_cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log_cfg.ini")
if not _config.init_from(_cfg_file):
    # raise Exception(f"Failed to read config from: {_cfg_file}")
    _config._config["LOGGING"]["log_level"] = logging.INFO
init_logger(_config)
