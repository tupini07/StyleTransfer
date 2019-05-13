import logging

import tqdm


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


def setup(level=logging.DEBUG):
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(module)s.%(funcName)s #%(lineno)d - %(message)s',
        handlers=[
            TqdmLoggingHandler()
        ]
    )
