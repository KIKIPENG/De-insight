"""統一錯誤處理工具。

提供 log_errors 裝飾器，取代所有 `except: pass` 的靜默吞錯行為。
"""

import functools
import logging
from typing import Callable, TypeVar

log = logging.getLogger("de-insight")

F = TypeVar("F", bound=Callable)


def log_errors(
    *,
    fallback=None,
    notify: bool = False,
    msg: str = "",
    level: int = logging.WARNING,
):
    """裝飾器：捕捉例外並結構化 log，取代 except: pass。

    Parameters
    ----------
    fallback : Any
        例外時的回傳值（預設 None）。
    notify : bool
        是否嘗試呼叫 self.notify()（僅 Mixin 方法可用）。
    msg : str
        自訂 log 訊息前綴。
    level : int
        logging level（預設 WARNING）。
    """

    def decorator(fn: F) -> F:
        if _is_async(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    prefix = msg or fn.__qualname__
                    log.log(level, "%s failed: %s", prefix, e, exc_info=True)
                    if notify and args and hasattr(args[0], "notify"):
                        try:
                            args[0].notify(
                                f"{prefix} 失敗: {e}" if not msg else f"{msg}: {e}",
                                severity="warning",
                                timeout=4,
                            )
                        except Exception:
                            pass
                    return fallback

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    prefix = msg or fn.__qualname__
                    log.log(level, "%s failed: %s", prefix, e, exc_info=True)
                    if notify and args and hasattr(args[0], "notify"):
                        try:
                            args[0].notify(
                                f"{prefix} 失敗: {e}" if not msg else f"{msg}: {e}",
                                severity="warning",
                                timeout=4,
                            )
                        except Exception:
                            pass
                    return fallback

            return sync_wrapper  # type: ignore

    return decorator


def _is_async(fn: Callable) -> bool:
    import asyncio

    return asyncio.iscoroutinefunction(fn)
