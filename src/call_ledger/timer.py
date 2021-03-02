import functools
import time

DURATION_STATS = {}

def timeit(func):
    """Gather stastistic on the decorated function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        started_at = time.perf_counter()
        func_name = func.__name__
        duration_key = 'duration'
        counter_key = 'counter'
        result = func(*args, **kwargs)
        ended_at = time.perf_counter()
        duration_of_call = ended_at - started_at
        stats = DURATION_STATS.get(func_name, {duration_key: 0, counter_key: 0})
        stats[duration_key] = stats[duration_key] + duration_of_call
        stats[counter_key] = stats[counter_key] + 1
        DURATION_STATS[func_name] = stats
        return result
    return wrapper

