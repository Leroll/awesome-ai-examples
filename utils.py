from time import time

def time_cost(func):
    def wraps(*args, **kargs):
        t0 = time()
        result = func(*args, **kargs)
        print(f'[{func.__name__}] cost time: {time() - t0:0.3f}s')
        return result
    return wraps


