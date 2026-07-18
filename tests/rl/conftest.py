"""Allow the copied async dispatcher tests to run in minimal test envs."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect


_HAS_PYTEST_ASYNCIO = importlib.util.find_spec("pytest_asyncio") is not None


def pytest_configure(config) -> None:
    if not _HAS_PYTEST_ASYNCIO:
        config.addinivalue_line("markers", "asyncio: run an async test with asyncio.run")


def pytest_pyfunc_call(pyfuncitem):
    if _HAS_PYTEST_ASYNCIO or not inspect.iscoroutinefunction(pyfuncitem.obj):
        return None
    arguments = {
        name: pyfuncitem.funcargs[name]
        for name in pyfuncitem._fixtureinfo.argnames
    }
    asyncio.run(pyfuncitem.obj(**arguments))
    return True
