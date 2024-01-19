import logging
import logging.config
import os
import sys
from functools import wraps

def init_logging(logging_path='../../logging.ini'):
    if os.path.exists(logging_path):
        logging.info(f'Loading logging file from {logging_path}')
        logging.config.fileConfig(fname=logging_path, disable_existing_loggers=False)
        logger = logging.getLogger()
        return logger
    else:
        logging.info(f'Logging file not found at {logging_path}')
        sys.exit(1)

def use_logger(logger):
    @wraps(logger)
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 在这里使用传入的logger对象
            logger.info(f'Calling function: {func.__name__}()')
            return func(*args, **kwargs)
        return wrapper
    return decorator


