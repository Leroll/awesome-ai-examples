from time import time


def call_time_with_name(name):
    """
    计时装饰器
    """
    def call_time(func):
        def wraps(*args, **kargs):
            t0 = time()
            result = func(*args, **kargs)
            print(f'{name}用时: {time() - t0:0.3f}')
            return result
        return wraps
    return call_time
