
"""
计时包装器
"""
import time

def time_cost(func):
    def wrapper(*arg, **kargs):
        t0 = time()
        res = func(*arg, **kargs)
        t1 = time()
        print(f'[{func.__name__}] cost {t1 - t0:.2f}s')
        return res
    return wrapper