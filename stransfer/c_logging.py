import logging
import os

import tqdm

from stransfer import constants

LOGGER = logging.getLogger('StyleTransfer')
LOGGER.setLevel(logging.DEBUG)
LOGGER.handlers = []

LOGGER_FORMATTER = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(module)s.%(funcName)s #%(lineno)d - %(message)s'
)


class TqdmLoggingHandler (logging.StreamHandler):
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


os.makedirs(constants.RUNS_PATH, exist_ok=True)
file_handler = logging.FileHandler(constants.LOG_PATH, mode='w+')
file_handler.setFormatter(LOGGER_FORMATTER)

LOGGER.addHandler(tqdm_handler)
LOGGER.addHandler(file_handler)


def get_logger() -> logging.Logger:
    return LOGGER
