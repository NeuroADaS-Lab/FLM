from __future__ import annotations
import logging
import time

def setup_loggers(*logger_names: str | logging.Logger, verbosity: str, log_file: str, console_verbosity: str = None):
    console_verbosity = console_verbosity or verbosity

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(filename=log_file)
    fh.setFormatter(formatter)
    fh.setLevel(verbosity)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(console_verbosity)

    for logger in logger_names:
        if isinstance(logger, str):
            logger = logging.getLogger(logger)

        # to avoid having a thousand fucking handlers
        if not logger.hasHandlers():
            logger.setLevel(verbosity)
            logger.addHandler(fh)
            logger.addHandler(ch)


def timestampify(root: str = "") -> str:
    timestamp = time.strftime("%d%m%Y_%H%M%S")
    if len(root) and not root.endswith('_'):
        root += '_'
    return root + timestamp
