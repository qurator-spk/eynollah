from multiprocessing import shared_memory
from contextlib import contextmanager
from functools import wraps
import numpy as np

@contextmanager
def share_ndarray(array: np.ndarray):
    size = np.dtype(array.dtype).itemsize * np.prod(array.shape)
    shm = shared_memory.SharedMemory(create=True, size=size)
    try:
        shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        shared_array[:] = array[:]
        shared_array.flags["WRITEABLE"] = False
        yield dict(shape=array.shape, dtype=array.dtype, name=shm.name)
    finally:
        shm.close()
        shm.unlink()

@contextmanager
def ndarray_shared(array: dict):
    shm = shared_memory.SharedMemory(name=array['name'])
    try:
        array = np.ndarray(array['shape'], dtype=array['dtype'], buffer=shm.buf)
        yield array
    finally:
        shm.close()

def wrap_ndarray_shared(kw=None):
    def wrapper(f):
        if kw is None:
            @wraps(f)
            def shared_func(array, *args, **kwargs):
                with ndarray_shared(array) as ndarray:
                    return f(ndarray, *args, **kwargs)
            return shared_func
        else:
            @wraps(f)
            def shared_func(*args, **kwargs):
                array = kwargs.pop(kw)
                with ndarray_shared(array) as ndarray:
                    kwargs[kw] = ndarray
                    return f(*args, **kwargs)
            return shared_func
    return wrapper

