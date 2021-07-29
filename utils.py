from time import time
from loguru import logger


def time_cost(func):
    def wraps(*args, **kargs):
        t0 = time()
        result = func(*args, **kargs)
        print(f'[{func.__name__}] cost time: {time() - t0:0.3f}s')
        return result
    return wraps


class LogGetter:
    def __init__(self):
        self.logger = logger
        self.logger.add('log.log',
                        enqueue=True,
                        colorize=True, format="<green>{time}</green> <level>{message}</level>",
                        level="INFO"
                        )

    def get_logger(self):
        return self.logger.info


log = LogGetter().get_logger()

if __name__ == '__main__':
    log('log test')



