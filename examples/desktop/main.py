# -*- coding: utf-8 -*-
"""Desktop application backend entry point."""
import argparse
import os

import uvicorn

from agentscope.app import create_app
from agentscope.app.deps import get_current_user_id
from agentscope.app.message_bus import InMemoryMessageBus
from agentscope.app.storage import RedisStorage
from agentscope.app.workspace_manager import LocalWorkspaceManager

DESKTOP_USER_ID = "local-user"
DATA_DIR = os.path.expanduser("~/.agentscope")

storage = RedisStorage(host="localhost", port=6379)

app = create_app(
    storage=storage,
    message_bus=InMemoryMessageBus(),
    workspace_manager=LocalWorkspaceManager(
        basedir=os.path.join(DATA_DIR, "workspaces"),
    ),
    enable_index_worker=True,
)


async def desktop_user_id() -> str:
    """Return fixed user ID for single-user desktop mode."""
    return DESKTOP_USER_ID


app.dependency_overrides[get_current_user_id] = desktop_user_id

if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )
    _args = _parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=_args.port)
