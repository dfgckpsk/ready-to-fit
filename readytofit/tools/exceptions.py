from .logging import get_logger


def try_catch(func):

    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException as e:
            logger = get_logger('try_catch_decorator')
            logger.error(str(e))

    return inner_function
