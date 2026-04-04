import asyncio

# ══════════════════════════════════════════════════════════════════════
# Python 3.14 / anyio compatibility shims
#
# Root cause: anyio's CancelScope and related primitives look up the
# current asyncio Task in a WeakKeyDictionary (_task_states). In Python
# 3.14, asyncio.current_task() returns None in two situations:
#
#   1. During connection-pool cleanup (teardown path) — no live task
#   2. Tasks spawned via asyncio.gather() outside anyio's task runner
#      — task exists but was never registered in _task_states
#
# Both cases crash with:
#   TypeError: cannot create weak reference to 'NoneType' object
# ══════════════════════════════════════════════════════════════════════


# ── Patch 1: Intercept anyio CancelScope.__enter__ ────────────────────
try:
    import anyio._backends._asyncio as _anyio_asyncio

    _task_states = _anyio_asyncio._task_states
    _TaskState = _anyio_asyncio.TaskState
    _OriginalCancelScope = _anyio_asyncio.CancelScope
    _original_cs_enter = _OriginalCancelScope.__enter__
    _original_cs_exit = _OriginalCancelScope.__exit__

    def _patched_cs_enter(self):
        host_task = asyncio.current_task()

        if host_task is None:
            # No live task at all (teardown path) — skip the scope entirely
            self.__dict__['_py314_noop'] = True
            return self

        if host_task not in _task_states:
            # Live task but not registered by anyio (spawned via asyncio.gather)
            # Register it with a minimal TaskState so anyio can proceed
            _task_states[host_task] = _TaskState(
                cancel_scope=None,
                asynclib_task_name_dict={},
            )

        self.__dict__['_py314_noop'] = False
        return _original_cs_enter(self)

    def _patched_cs_exit(self, *args):
        if self.__dict__.get('_py314_noop', False):
            return False  # did not suppress exception
        return _original_cs_exit(self, *args)

    _OriginalCancelScope.__enter__ = _patched_cs_enter
    _OriginalCancelScope.__exit__ = _patched_cs_exit

except (ImportError, AttributeError) as e:
    import warnings
    warnings.warn(f"sniffio_compat: CancelScope patch failed: {e}")


# ── Patch 2: sniffio AsyncLibraryNotFoundError fallback ───────────────
# When tasks are not registered with sniffio, default to "asyncio".
try:
    import sniffio
    import sniffio._impl as _sniffio_impl

    _original_current_async_library = sniffio.current_async_library

    def _safe_current_async_library():
        try:
            return _original_current_async_library()
        except sniffio.AsyncLibraryNotFoundError:
            try:
                asyncio.get_running_loop()
                return "asyncio"
            except RuntimeError:
                pass
            raise

    sniffio.current_async_library = _safe_current_async_library
    _sniffio_impl.current_async_library = _safe_current_async_library

except ImportError:
    pass


def apply():
    """No-op: patches are applied at import time. Call this to make intent explicit."""
    pass