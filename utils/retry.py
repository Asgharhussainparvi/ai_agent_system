"""utils/retry.py — Exponential-backoff retry decorator."""
import time, functools
from typing import Tuple, Type
from utils.logger import get_logger

_log = get_logger("Retry")


def with_retry(
    max_retries: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Decorator: retry `max_retries` times with exponential backoff."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            wait = delay
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_retries:
                        raise
                    _log.warning(
                        "Attempt %d/%d failed for '%s': %s — retrying in %.0fs…",
                        attempt + 1, max_retries, fn.__qualname__, exc, wait,
                    )
                    time.sleep(wait)
                    wait *= backoff
        return wrapper
    return decorator
