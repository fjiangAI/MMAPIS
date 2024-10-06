import logging
import logging.config
import os
from MMAPIS.backend.config.config import LOGGER_MODES,DIR_PATH

def init_logging(logging_dir=DIR_PATH):
    logging_path = os.path.join(logging_dir,"logging.ini")
    logging_path = os.path.abspath(logging_path)
    if os.path.exists(logging_path):
        logging.info(f'Loading logging file from {logging_path}')
        logging.config.fileConfig(fname=logging_path, disable_existing_loggers=False)
    else:
        logging.info(f'Logging file not found at {logging_path}')
        raise Exception(f"No logging config found in {logging_path}")



