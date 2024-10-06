from functools import wraps
import logging
def handle_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logging.error(f'{func.__name__} error: {e}')
            return
    return wrapper



