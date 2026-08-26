# -*- coding: utf-8 -*-
"""PyInstaller hook for AgentScope package data and dynamic drivers."""
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

datas = collect_data_files("agentscope")
datas += collect_data_files(
    "agentscope",
    include_py_files=True,
    includes=["app/storage/_sql/_alembic/**"],
)
datas += copy_metadata("qdrant-client")
hiddenimports = [
    "aiosqlite",
    "portalocker",
    "sqlalchemy.dialects.sqlite.aiosqlite",
]
