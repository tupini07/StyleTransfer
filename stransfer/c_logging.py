import logging

import tqdm

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


tqmd_handler = TqdmLoggingHandler()
tqmd_handler.setFormatter(LOGGER_FORMATTER)

file_handler = logging.FileHandler('data/runtime.log', mode='w+')
file_handler.setFormatter(LOGGER_FORMATTER)

LOGGER.addHandler(tqmd_handler)
LOGGER.addHandler(file_handler)


def get_logger() -> logging.Logger:
    return LOGGER
