"""
This module contains functionality for logging.
It mainly configures the log level and a handler
so that logging can coexist with the `tqdm` progress bar.
"""
import logging
import os

import tqdm

from stransfer import constants

# Setup default logger for the application
_LOGGER = logging.getLogger('StyleTransfer')

# Set logging level
_LOGGER.setLevel(logging.INFO)

# clear handlers
_LOGGER.handlers = []

LOGGER_FORMATTER = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(module)s.%(funcName)s #%(lineno)d - %(message)s'
)


class TqdmLoggingHandler(logging.StreamHandler):
    """
    Custom logging handler. This ensures that TQDM
    progress bars always stay at the bottom of the
    terminal instead of being printed as normal
    messages
    """

    def __init__(self, stream=None):
        super().__init__(stream)

    # we only overwrite `Logging.StreamHandler`
    # emit method
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


tqdm_handler = TqdmLoggingHandler()
tqdm_handler.setFormatter(LOGGER_FORMATTER)

# we need to ensure that the runs_path exists since there is
# where we'll save a log file of the execution
os.makedirs(constants.RUNS_PATH, exist_ok=True)
file_handler = logging.FileHandler(constants.LOG_PATH, mode='w+')
file_handler.setFormatter(LOGGER_FORMATTER)

_LOGGER.addHandler(tqdm_handler)
_LOGGER.addHandler(file_handler)


def get_logger() -> logging.Logger:
    """
    :return: the global loader for the application
    """
    return _LOGGER
