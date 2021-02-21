import functools
import time

DURATION_STATS = {}

def timeit(func):
    """Gather stastistic on the decorated function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        started_at = time.perf_counter()
        func_name = func.__name__
        duration_key = func_name+'.duration'
        counter_key = func_name+'.counter'
        result = func(*args, **kwargs)
        ended_at = time.perf_counter()
        duration_of_call = ended_at - started_at
        DURATION_STATS[duration_key] = DURATION_STATS.get(duration_key, 0) + duration_of_call
        DURATION_STATS[counter_key] = DURATION_STATS.get(counter_key, 0) + 1
        return result
    return wrapper

