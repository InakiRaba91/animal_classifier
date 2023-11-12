from contextlib import contextmanager


@contextmanager
def does_not_raise():
    """context manager for when we expect no error raised"""
    yield
